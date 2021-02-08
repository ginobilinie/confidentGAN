import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
from torchvision import transforms
from transformers import Rescale
import os
import os.path as osp
import pickle
from config import config
from tqdm import tqdm
from dataloader import MyDataSet
from torch.utils import data
from blocks import PolyLR, init_weight
from sync_batchnorm import SynchronizedBatchNorm2d
from generator import NestedUNet
from discriminator import Discriminator_FCN
from losses import ProbOhemCrossEntropy2d, Dice_loss
from lovasz_losses import lovasz_softmax
#import matplotlib.pyplot as plt
import random
import timeit
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"  # specify which GPU(s) to be used



def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(config.lr, i_iter, config.total_niters, config.lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(config.lr_D, i_iter, config.total_niters, config.lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def one_hot(label):
    label = label.cpu().numpy()
    one_hot = np.zeros((label.shape[0], config.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(config.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot).cuda()

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)

    return D_label


def main():
    if not os.path.exists(config.snapshot_dir):
        os.makedirs(config.snapshot_dir)

    BatchNorm2d = SynchronizedBatchNorm2d

    # loss for adversarial training
    bce_criterion = nn.BCELoss().cuda()

    # loss for segmentation
    # criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=100000, use_weight=False)
    dice_criterion = Dice_loss(config.num_classes)
    # dice_criterion = lovasz_softmax

    # create network
    model = NestedUNet(config.num_classes, criterion=criterion,
                dice_criterion=dice_criterion, 
                is_training=True,
                norm_layer=BatchNorm2d,
                gamma=2) #Note gamma is the hyper-parameter for adversarial confidence learning
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.train()

    # init D
    model_D = Discriminator_FCN(num_classes=config.num_classes+3)
    init_weight(model_D.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum)
    model_D = model_D.cuda()
    model_D = nn.DataParallel(model_D, device_ids=range(torch.cuda.device_count()))

    if config.restore_from_D is not None:
        model_D.load_state_dict(torch.load(config.restore_from_D))
    model_D.train()

    train_dataset = MyDataSet(config.train_source, img_mean=config.image_mean, img_std=config.image_std, transform=transforms.Compose([Rescale((1024, 1024))]))
    # val_dataset = MyDataSet(config.eval_source)
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    # val_loader = data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    # optimizer for segmentation network
    optimizer = optim.SGD(model.parameters(),
                lr=config.lr, momentum=config.momentum,weight_decay=config.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.SGD(model_D.parameters(), lr=config.lr_D, momentum=config.momentum,weight_decay=config.weight_decay)
    optimizer_D.zero_grad()


    base_lr = config.lr
    # config lr policy

    for epoch in range(config.nepochs):
        # t = tqdm(enumerate(iter(train_loader)), leave=False, total=len(train_loader))
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)
        # for batch_idx, batch in t:
        for batch_idx in pbar:
            minibatch = dataloader.next()
            x_ = Variable(minibatch[0]).type(torch.FloatTensor).contiguous().cuda(non_blocking=True)
            y_ = Variable(minibatch[1]).type(torch.LongTensor).contiguous().cuda(non_blocking=True)

            current_idx = epoch * config.niters_per_epoch + batch_idx

            # train D first
            optimizer_D.zero_grad()
            lr_D = adjust_learning_rate_D(optimizer_D, current_idx)

            x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())

            D_result = model_D(x_, one_hot(y_))
            # print('D_result.shape is ',D_result.shape)
            D_result = F.interpolate(D_result, size=(x_.shape[-1], x_.shape[-2]), mode='bilinear', align_corners=True).squeeze()
            # print('D_result.shape is ',D_result.shape)
            #print('shape0: ',D_result.shape, '...', D_result, '...')
            D_result = F.sigmoid(D_result)
            #print('shape1: ',D_result.squeeze().shape, '...', D_result, '...')
            #cv2.imwrite('real_output.jpg', D_result[0, ...].cpu().detach().numpy().squeeze())
            D_real_loss = bce_criterion(D_result, Variable(torch.ones(D_result.size()).cuda()))

            G_result = model(x_) # has already softmaxed
            D_result = model_D(x_, G_result)
            D_result = F.interpolate(D_result, size=(x_.shape[-1], x_.shape[-2]), mode='bilinear', align_corners=True).squeeze()
            D_result = F.sigmoid(D_result)
            #cv2.imwrite('fake_output.jpg',D_result[0,...].cpu().detach().numpy().squeeze())
            D_fake_loss = bce_criterion(D_result, Variable(torch.zeros(D_result.size()).cuda()))
            #print('D_real_loss.mean: ',D_real_loss.mean().item(),' D_fake_loss.mean(): ',D_fake_loss.mean().item())
            D_train_loss = (D_real_loss + D_fake_loss) * 0.5
            D_train_loss = D_train_loss.mean()
            D_train_loss.backward()
            optimizer_D.step()


            # then train G
            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, current_idx)
            D_result = model_D(x_, G_result)
            D_result = F.interpolate(D_result, size=(x_.shape[-1], x_.shape[-2]), mode='bilinear', align_corners=True).squeeze()
            D_result = F.sigmoid(D_result)
            G_adv_loss = bce_criterion(D_result, Variable(torch.ones(D_result.size()).cuda()))
            G_result, loss_seg, loss_ce, loss_dice = model(x_,y_, D_result)
            loss_seg = loss_seg.mean()
            G_adv_loss = G_adv_loss.mean()
            G_train_loss = config.lambda_adv*G_adv_loss + loss_seg
            G_train_loss.backward()
            optimizer.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(batch_idx + 1, config.niters_per_epoch) \
                        + ' lr_D=%.3e' %lr_D \
                        + ' D_train_loss=%.2f'%D_train_loss.item()\
                        + ' lr=%.3e' % lr \
                        + ' G_adv_loss=%.2f' % G_adv_loss.item() \
                        + ' loss_seg=%.2f' % loss_seg.item() \
                        + ' loss_ce=%.2f' % loss_ce.mean().item() \
                        + ' loss_dice=%.2f' % loss_dice.mean().item() +'\n'

            pbar.set_description(print_str, refresh=False)

        if (epoch > config.nepochs - 20) or (epoch % config.snapshot_iter == 0):
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(config.snapshot_dir,'modelG_Epoch%d.pth'%epoch))
            stateD = {
                'epoch': epoch,
                'state_dict': model_D.state_dict(),
                'optimizer': optimizer_D.state_dict(),
            }
            torch.save(stateD, os.path.join(config.snapshot_dir,'modelD_Epoch%d.pth'%epoch))


if __name__ == '__main__':
    main()

import cv2
import os, sys
import numpy as np

import torch
import torch.utils.data
from generator import NestedUNet 
from dataloader import MyDataSet
from losses import Dice_loss, ProbOhemCrossEntropy2d
#from dice import DiceLoss
from config import config
#import tools

#pb = tools.pb
import torch.nn as nn
from torch.utils.data import DataLoader
from eval import compute_iou

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

seed = 304
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def load_model(model, model_file, is_restore=False):
    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file

    #     if is_restore:
    #         new_state_dict = OrderedDict()
    #         for k, v in state_dict.items():
    #             name = 'module.' + k
    #             new_state_dict[name] = v
    #         state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    del state_dict
    return model


BATCH_SIZE = 1

val_dataset = MyDataSet(config.eval_source)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

base_lr = 3e-2
criterion = nn.CrossEntropyLoss(reduction='mean',
                                ignore_index=255)
ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
                                        min_kept=int(
                                            48 // 8 * 1024 * 1024 // 16),
                                        use_weight=False)

BatchNorm2d = nn.BatchNorm2d

model = NestedUNet(config.num_classes, is_training=False,
              criterion=ohem_criterion,
              dice_criterion=None,
              norm_layer=BatchNorm2d)

print(model)

print('test iou')

epoch=299
model_path = os.path.join(config.snapshot_dir,'modelG_Epoch%d.pth'%epoch)

# correctly saved model without DataParallel or DDP
checkpoint = torch.load(model_path)
state_dict = checkpoint['state_dict']

# if it is stored without calling xx.module.state_dict, then we have to do the following stuff
from collections import OrderedDict
new_state_dict = OrderedDict()
for k,v in state_dict.items():
    name = k[7:] # remove module
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
# model.load_state_dict(state_dict)
model.eval()
model = model.cuda()

mean_iou = compute_iou(model, val_loader, config.num_workers, 1024, BATCH_SIZE,  epoch)
print(mean_iou)
print('end')


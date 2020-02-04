# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.checkpoint import checkpoint
# from torchvision.models import resnet50, resnet101, resnet152

from config import config
# from base_model import resnet101
from blocks import ConvBnRelu



class UNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale, norm_layer=nn.BatchNorm2d):
        super(UNetHead, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = nn.Conv2d(64,out_planes, kernel_size=1, stride=1, padding=0)

        self.scale = scale

    def forward(self, x):
        x = self.conv_3x3(x)
        x = self.conv_1x1(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear',
                          align_corners=True)

        return x


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, out_planes, criterion, dice_criterion=None, is_training=True, norm_layer=nn.BatchNorm2d):
        super(NestedUNet, self).__init__()

        # self.args = args
        self.is_training = is_training

        self.layer0 = ConvBnRelu(3, 64, ksize=7, stride=2, pad=3, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.criterion = criterion
        self.dice_criterion = dice_criterion
        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])


        self.heads = [UNetHead(64, out_planes, 4, norm_layer=norm_layer),
                UNetHead(64, out_planes, 4,  norm_layer=norm_layer),
                UNetHead(64, out_planes, 4,  norm_layer=norm_layer),
                UNetHead(64, out_planes, 4,  norm_layer=norm_layer),
        ]
        self.heads = nn.ModuleList(self.heads)
        self.business_layer = []
        self.business_layer.append(self.layer0)
        self.business_layer.append(self.conv0_0)
        self.business_layer.append(self.conv1_0)
        self.business_layer.append(self.conv2_0)
        self.business_layer.append(self.conv3_0)
        self.business_layer.append(self.conv4_0)

        self.business_layer.append(self.conv0_1)
        self.business_layer.append(self.conv1_1)
        self.business_layer.append(self.conv2_1)
        self.business_layer.append(self.conv3_1)

        self.business_layer.append(self.conv0_2)
        self.business_layer.append(self.conv1_2)
        self.business_layer.append(self.conv2_2)

        self.business_layer.append(self.conv0_3)
        self.business_layer.append(self.conv0_4)

        self.business_layer.append(self.heads)


        # if self.args.deepsupervision:
        #     self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        #     self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        #     self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        #     self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        # else:
        #     self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)


    def forward(self, input, label=None):
        x = self.layer0(input)
        x = self.maxpool(x)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.is_training and label is not None:
            loss0 = self.criterion(self.heads[0](x0_1), label)
            loss1 = self.criterion(self.heads[1](x0_2), label)
            loss2 = self.criterion(self.heads[2](x0_3), label)
            loss3 = self.criterion(self.heads[3](x0_4), label)

            dice_loss0 = self.dice_criterion(self.heads[0](x0_1), label)
            dice_loss1 = self.dice_criterion(self.heads[1](x0_2), label)
            dice_loss2 = self.dice_criterion(self.heads[2](x0_3), label)
            dice_loss3 = self.dice_criterion(self.heads[3](x0_4), label)

            loss = loss0 + loss1 + loss2 + loss3
            dice_loss = dice_loss0 + dice_loss1 + dice_loss2 + dice_loss3

            return self.heads[-1](x0_4), loss+dice_loss, loss, dice_loss

        output = self.heads[-1](x0_4)
        return F.softmax(output, dim=1)

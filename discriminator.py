import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.checkpoint import checkpoint
# from torchvision.models import resnet50, resnet101, resnet152

# from config import config
# from base_model import resnet101
# from seg_opr.seg_oprs import ConvBnRelu
from blocks import ConvBnRelu


class Discriminator_CNN(nn.Module):
    # initializers
    def __init__(self, input_channels, d=64, norm_layer=nn.BatchNorm2d):
        super(Discriminator_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, d, 4, 2, 1)
        self.conv2_bn = nn.Sequential(nn.Conv2d(d, d * 2, 4, 2, 1),  norm_layer(d * 2))
        # self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        # self.conv2_bn = norm_layer(d * 2)
        self.conv3_bn = nn.Sequential(nn.Conv2d(d * 2, d * 4, 4, 2, 1), norm_layer(d * 4))
        # self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        # self.conv3_bn = norm_layer(d * 4)
        self.conv4_bn = nn.Sequential(nn.Conv2d(d * 4, d * 8, 4, 1, 1), norm_layer(d * 8))
        # self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        # self.conv4_bn = norm_layer(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

        self.business_layer = []
        self.business_layer.append(self.conv1)
        self.business_layer.append(self.conv2_bn)
        self.business_layer.append(self.conv3_bn)
        self.business_layer.append(self.conv4_bn)
        self.business_layer.append(self.conv5)
    # # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(x), 0.2)
        x = F.leaky_relu(self.conv3_bn(x), 0.2)
        x = F.leaky_relu(self.conv4_bn(x), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


class Discriminator_FCN(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(Discriminator_FCN, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(ndf*8, ndf*4, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*4, 1, kernel_size=3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()
        self.business_layer = []
        self.business_layer.append(self.conv1)
        self.business_layer.append(self.conv2)
        self.business_layer.append(self.conv3)
        self.business_layer.append(self.conv4)
        self.business_layer.append(self.deconv4)
        self.business_layer.append(self.classifier)

    def forward(self, x, label):
        #print('x.shape:',x.shape,'label.shape:',label.shape)
        x = torch.cat((x,label),1)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.deconv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        #x = self.up_sample(x)
        #x = self.sigmoid(x)

        return x

# coding: utf-8


import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from config import config
import random
from img_utils import random_scale,random_mirror,generate_random_crop_pos,random_crop_pad_to_shape

class MyDataSet(Dataset):
    def __init__(self, dataset, img_mean=None, img_std= None, transform=None):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self.transform = transform
        self.img_mean = img_mean
        self.img_std = img_std

        with open(dataset, 'r') as file:
            for _info in file:
                info_tmp = _info.strip().split(' ')

                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])

        assert len(self._gt_img_list) == len(self._gt_label_binary_list)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self._gt_img_list, self._gt_label_binary_list))
        random.shuffle(c)
        self._gt_img_list, self._gt_label_binary_list = zip(*c)

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        assert len(self._gt_label_binary_list) == len(self._gt_img_list)

        # load all
        #print(config.dataset_path+self._gt_img_list[idx])
        #print(config.dataset_path)
        img = cv2.imread(config.dataset_path+self._gt_img_list[idx], cv2.IMREAD_COLOR)
        img = img[:,:,::-1] #bgr to rgb

        label_img = cv2.imread(config.dataset_path+self._gt_label_binary_list[idx], cv2.IMREAD_GRAYSCALE)


        # optional transformations
        if self.transform:
            img = self.transform(img)
            label_img = self.transform(label_img)
        ## more transformations
        img, label_img = random_mirror(img, label_img)
        if config.train_scale_array is not None:
            img, label_img, scale = random_scale(img, label_img, config.train_scale_array)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        label_img, _ = random_crop_pad_to_shape(label_img, crop_pos, crop_size, 255)

        # extract each label into separate binary channels

        # reshape for pytorch
        # tensorflow: [height, width, channels]
        # pytorch: [channels, height, width]
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        img = img/255.0
        if self.img_mean is not None:
            img[0,...] = img[0,...] - self.img_mean[0]
            img[1,...] = img[1,...] - self.img_mean[1]
            img[2,...] = img[2,...] - self.img_mean[2]

        # we could split the instance label here, each instance in one channel (basically a binary mask for each)
        return (img, label_img)

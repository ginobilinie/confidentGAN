import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from data import PredictFlow, test_one_img, colorize, test_batch_img


EPSILON = 1e-7


def compute_iou(net, data_loader, num_classes, img_size, batch_size, pb, epoch):
    net.eval()
    data_loader_iter = iter(data_loader)
    pb.reset(len(data_loader_iter))
    intersections, unions = [], []
    for index, (img, label_mask) in enumerate(data_loader_iter):

        print(index)
        label_mask = label_mask.cpu().data.numpy()
        probs = test_one_img(net, img)
        predict_mask = np.argmax(probs, axis=0)
        ious_per_image = []
        intersections_per_image, unions_per_image = [], []
        for class_id in range(0, num_classes):
            predict_mask_per_class = (predict_mask == class_id).astype('float32')
            label_mask_per_class = (label_mask == class_id).astype('float32')

            area_predict_mask_per_class = np.sum(predict_mask_per_class)
            area_label_mask_per_class = np.sum(label_mask_per_class)

            intersection = np.sum(predict_mask_per_class * label_mask_per_class)
            union = area_predict_mask_per_class + area_label_mask_per_class - intersection

            intersections_per_image.append(intersection)
            unions_per_image.append(union)

            iou = intersection / (union + EPSILON)
            ious_per_image.append(iou)

        intersections.append(intersections_per_image)
        unions.append(unions_per_image)
        pb.show(0, 0, index, "iou: {:.4f}".format(np.mean(ious_per_image)))

    intersections_by_class = np.sum(intersections, axis=0)
    unions_by_class = np.sum(unions, axis=0)
    ious_by_class = intersections_by_class / (unions_by_class + EPSILON)

    mean_iou = np.mean(ious_by_class)
    msg = "epoch {} iou:{} mean_iou:{:.3f}".format(epoch, ious_by_class, mean_iou)
    pb.summary(msg)
    return mean_iou
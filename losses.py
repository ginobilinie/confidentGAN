import numpy as np
import scipy.ndimage as nd

import torch
import torch.nn as nn
import torch.nn.functional as F

# from engine.logger import get_logger
# from seg_opr.seg_oprs import one_hot

# logger = get_logger()


class SigmoidFocalLoss(nn.Module):
    def __init__(self, ignore_label, gamma=2.0, alpha=0.25,
                 reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, h, w = target.size()
        pred = pred.view(b, -1, 1)
        pred_sigmoid = pred.sigmoid()
        target = target.view(b, -1).float()
        mask = (target.ne(self.ignore_label)).float()
        target = mask * target
        onehot = target.view(b, -1, 1)

        # TODO: use the pred instead of pred_sigmoid
        max_val = (-pred_sigmoid).clamp(min=0)

        pos_part = (1 - pred_sigmoid) ** self.gamma * (
                pred_sigmoid - pred_sigmoid * onehot)
        neg_part = pred_sigmoid ** self.gamma * (max_val + (
                (-max_val).exp() + (-pred_sigmoid - max_val).exp()).log())

        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(
            dim=-1) * mask
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [1.4297, 1.4805, 1.4363, 3.365, 2.6635, 1.4311, 2.1943, 1.4817,
                 1.4513, 2.1984, 1.5295, 1.6892, 3.2224, 1.4727, 7.5978, 9.4117,
                 15.2588, 5.6818, 2.2067])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            # logger.info('Labels: {}'.format(num_valid))
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = torch.sort(mask_prob)
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

class Dice_loss:
    def __init__(self, dice_weight=0, class_weights=None, num_classes=1):
        super(Dice_loss, self).__init__()
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.class_weights = class_weights
    def __call__(self, outputs, targets):
        loss_dice = 0
        smooth = 1.
        outputs = F.softmax(outputs, dim=1)
        for cls in range(self.num_classes):
            jaccard_target = (targets == cls).float()
            jaccard_output = outputs[:, cls]
            intersection = (jaccard_output * jaccard_target).sum()
            if self.class_weights is not None:
                w = self.class_weights[cls]
            else:
                w = 1.
            union = jaccard_output.sum() + jaccard_target.sum()
#                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
            loss_dice += w*(1- (2.*intersection + smooth) / (union  + smooth))
            # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return loss_dice*self.num_classes
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
        min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

class WeightedMSE(nn.Module):
    def __init__(self, weight=None):
        super(WeightedMSE, self).__init__()
        self.weight = weight

    def forward(self, score, target):
        score = score.float()
        target = target.float()

        if self.weight is not None:
            eps = 1e-5

            # calc mse with target label of 1
            tmp_score = torch.where(
                target == 1,
                score,
                torch.zeros(score.shape, dtype=torch.float32).cuda()
            )
            loss_label1 = 0.5 * torch.sum((1 - self.weight) * (tmp_score - target) ** 2, dim=0)

            # calc mse with target label of 0
            tmp_score = torch.where(
                target == 0,
                score,
                torch.ones(score.shape, dtype=torch.float32).cuda()
            )
            loss_label0 = 0.5 * torch.sum(self.weight * (tmp_score - target) ** 2, dim=0)

            return loss_label1 + loss_label0
        else:
            return 0.5 * torch.sum((score - target) ** 2, dim=0)

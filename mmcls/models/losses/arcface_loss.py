# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from mmcv.runner import force_fp32
from torch import nn
from torch.nn import functional as F

from mmcls.models.losses.cross_entropy_loss import soft_cross_entropy
from mmcls.registry import MODELS


@MODELS.register_module()
class ArcFaceLoss(nn.Module):

    def __init__(self,
                 s: float = 30.0,
                 m: float = 0.5,
                 reduction='mean',
                 loss_weight: float = 1.0,
                 class_weight=None):
        super().__init__()
        self.s = s
        self.m = m
        self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)  # (t + m) == pi

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.fp16_enabled = False

    @force_fp32(apply_to=('input', ))
    def forward(self,
                input,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        with torch.cuda.amp.autocast(enabled=False):
            if self.class_weight is not None:
                class_weight = input.new_tensor(self.class_weight)
            else:
                class_weight = None

            with torch.no_grad():
                ont_hot_target = F.one_hot(target, num_classes=input.size(-1))

            cos_t = input.clamp(-1, 1)
            cos_t_m = torch.cos(torch.acos(cos_t) + self.m)
            cos_t_m = torch.where(cos_t > self.threshold, cos_t_m,
                                  cos_t - self.mm)

            logit = ont_hot_target * cos_t_m + (1 - ont_hot_target) * cos_t
            logit = logit * self.s

            loss = soft_cross_entropy(
                logit,
                ont_hot_target,
                weight=weight,
                reduction=reduction,
                class_weight=class_weight,
                avg_factor=avg_factor)
        return self.loss_weight * loss

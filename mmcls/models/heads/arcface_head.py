# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS, build_loss
from ..utils import is_tracing
from .base_head import BaseHead


@HEADS.register_module()
class ArcFaceHead(BaseHead):
    """ArcFace classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        s (float): Norm of input feature. Defaults to 30.0.
        m (float): Margin. Defaults to 0.5.
        easy_margin (bool): Avoid theta + m >= PI. Defaults to False.
        ls_eps (float): Label smoothing. Defaults to 0.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 s: float = 30.0,
                 m: float = 0.50,
                 easy_margin: bool = False,
                 ls_eps: float = 0.0,
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[dict] = None):
        super(ArcFaceHead, self).__init__(init_cfg=init_cfg)
        self.compute_loss = build_loss(loss)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_channels))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]

        warnings.warn(
            'The input of ClsHead should be already logits. '
            'Please modify the backbone if you want to get pre-logits feature.'
        )
        return x

    def forward_train(self, x, gt_label, **kwargs):
        pre_logits = self.pre_logits(x)
        if gt_label is None:
            return pre_logits

        cosine = F.linear(F.normalize(pre_logits), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=pre_logits.device)
        one_hot.scatter_(1, gt_label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 -
                       self.ls_eps) * one_hot + self.ls_eps / self.num_classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        losses = self.loss(output, gt_label, **kwargs)
        return losses

    def loss(self, cls_score: torch.Tensor, gt_label: torch.Tensor,
             **kwargs) -> dict:
        # compute loss
        losses = dict()
        loss = self.compute_loss(
            cls_score, gt_label, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        return losses

    def simple_test(self, feats, post_process=True):
        """Inference without augmentation.

        Args:
            feats: (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to retrieval. The shape of every item should be
                ``(num_samples, num_classes)``.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            torch.Tensor: features of samples.
        """

        pred = self.pre_logits(feats)
        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

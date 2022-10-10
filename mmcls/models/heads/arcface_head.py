# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from .cls_head import ClsHead


class NormLinear(nn.Linear):
    """An enhanced linear layer, which could normalize the input and the linear
    weight.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample
        bias (bool): Whether there is bias. If set to ``False``, the
            layer will not learn an additive bias. Defaults to ``True``.
        feature_norm (bool): Whether to normalize the input feature.
            Defaults to ``True``.
        weight_norm (bool):Whether to normalize the weight.
            Defaults to ``True``.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 feature_norm: bool = True,
                 weight_norm: bool = True):

        super().__init__(in_features, out_features, bias=bias)
        self.weight_norm = weight_norm
        self.feature_norm = feature_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_norm:
            input = F.normalize(input)
        if self.weight_norm:
            weight = F.normalize(self.weight)
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)


@MODELS.register_module()
class ArcFaceClsHead(ClsHead):
    """ArcFace classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        s (float): Norm of input feature. Defaults to 30.0.
        m (float): Margin. Defaults to 0.5.
        easy_margin (bool): Avoid theta + m >= PI. Defaults to False.
        ls_eps (float): Label smoothing. Defaults to 0.
        bias (bool): Whether to use bias in norm layer. Defaults to False.
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
                 bias: bool = False,
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[dict] = None):

        super(ArcFaceClsHead, self).__init__(init_cfg=init_cfg)
        self.loss_module = MODELS.build(loss)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.s = s
        self.m = m
        self.ls_eps = ls_eps

        self.norm_linear = NormLinear(in_channels, num_classes, bias=bias)

        self.easy_margin = easy_margin
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ArcFaceHead``, we just obtain the
        feature of the last stage.
        """
        # The ArcFaceHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self,
                feats: Tuple[torch.Tensor],
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The forward process."""

        pre_logits = self.pre_logits(feats)

        # cos=(a*b)/(||a||*||b||)
        cosine = self.norm_linear(pre_logits)

        if target is None:
            return self.s * cosine

        phi = torch.cos(torch.acos(cosine) + self.m)

        if self.easy_margin:
            # when cosine>0, choose phi
            # when cosine<=0, choose cosine
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # when cos>th, choose phi
            # when cos<=th, choose cosine-mm
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=pre_logits.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 -
                       self.ls_eps) * one_hot + self.ls_eps / self.num_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[ClsDataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_label.score for i in data_samples])
        else:
            target = torch.hstack([i.gt_label.label for i in data_samples])

        # The part can be traced by torch.fx
        cls_score = self(feats, target)

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        return losses

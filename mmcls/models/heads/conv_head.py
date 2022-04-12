# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class ConvClsHead(ClsHead):
    """Convlution 1x1 classifier head.

    Args:
        num_classes (int): Number of categories including the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type="Kaiming", layer=["Conv2d"], nonlinearity="leaky_relu").
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        init_cfg: Dict = dict(
            type="Kaiming", layer=["Conv2d"], nonlinearity="leaky_relu"
        ),
        act_cfg: Dict = None,
        norm_cfg: Dict = None,
        *args,
        **kwargs,
    ):
        super(ConvClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f"num_classes={num_classes} must be a positive integer"
            )

        self.conv = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels, X, X)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes, X, X)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes, X, X)``.
        """
        x = self.pre_logits(x)
        cls_score = self.conv(x)

        if softmax:
            cls_score = cls_score.view(x.size(0), -1)
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None
            )
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self.conv(x)
        cls_score = cls_score.view(x.size(0), -1)

        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

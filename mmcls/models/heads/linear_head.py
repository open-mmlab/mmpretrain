# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        lazy_linear (bool): If True, the fc layer use nn.LazyLinear.
            Defaults to False.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 lazy_linear=False,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lazy_linear = lazy_linear

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        if self.lazy_linear:
            warnings.warn('For LinearClsHead with lazy_linear=True, '
                          f'in_channels={in_channels} and '
                          f'init_cfg={self.init_cfg}  is ignored and '
                          f'calculated automatically.')
            self.fc = nn.LazyLinear(self.num_classes)
        else:
            self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        if not self.lazy_linear:
            super(LinearClsHead, self).init_weights()

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
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

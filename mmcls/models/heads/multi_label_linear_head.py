# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .multi_label_head import MultiLabelClsHead


@HEADS.register_module()
class MultiLabelLinearClsHead(MultiLabelClsHead):
    """Linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(MultiLabelLinearClsHead, self).__init__(
            loss=loss, init_cfg=init_cfg)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward_train(self, x, gt_label):
        if isinstance(x, tuple):
            x = x[-1]
        gt_label = gt_label.type_as(x)
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, x):
        """Test without augmentation."""
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = self.fc(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.sigmoid(cls_score) if cls_score is not None else None

        return self.post_process(pred)

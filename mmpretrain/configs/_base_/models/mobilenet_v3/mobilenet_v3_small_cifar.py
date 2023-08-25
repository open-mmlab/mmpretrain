# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from torch.nn.modules.activation import Hardswish

from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, MobileNetV3,
                               StackedLinearClsHead)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=MobileNetV3, arch='small'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=StackedLinearClsHead,
        num_classes=10,
        in_channels=576,
        mid_channels=[1280],
        act_cfg=dict(type=Hardswish),
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5)))

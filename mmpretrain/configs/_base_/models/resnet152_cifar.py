# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, ResNet_CIFAR)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=ResNet_CIFAR,
        depth=152,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=2048,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
    ))

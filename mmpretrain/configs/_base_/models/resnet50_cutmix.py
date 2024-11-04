# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmpretrain.models import (CrossEntropyLoss, CutMix, GlobalAveragePooling,
                               ImageClassifier, MultiLabelLinearClsHead,
                               ResNet)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=MultiLabelLinearClsHead,
        num_classes=1000,
        in_channels=2048,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0, use_soft=True)),
    train_cfg=dict(
        augments=dict(type=CutMix, alpha=1.0, num_classes=1000, prob=1.0)),
)

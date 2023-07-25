# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmpretrain.models import (VisionTransformer, ImageClassifier, VisionTransformerClsHead, CrossEntropyLoss)
from mmengine.model.weight_init import KaimingInit

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=VisionTransformer,
        arch='l',
        img_size=224,
        patch_size=32,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type=KaimingInit,
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type=VisionTransformerClsHead,
        num_classes=1000,
        in_channels=1024,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5),
    ))
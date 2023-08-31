# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, SwinTransformer)

# model settings
# Only for evaluation
model = dict(
    type=ImageClassifier,
    backbone=dict(type=SwinTransformer, arch='large', img_size=224),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=1536,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5)))

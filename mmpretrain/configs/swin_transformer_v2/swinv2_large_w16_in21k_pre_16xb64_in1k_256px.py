# Only for evaluation
# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

from mmpretrain.models import CrossEntropyLoss

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_256 import *
    from .._base_.default_runtime import *
    from .._base_.models.swin_transformer_v2_base import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

# model settings
model.update(
    backbone=dict(
        arch='large',
        img_size=256,
        window_size=[16, 16, 16, 8],
        pretrained_window_sizes=[12, 12, 12, 6]),
    head=dict(
        in_channels=1536,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5)))

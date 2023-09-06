# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base
from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import CutMix, Mixup

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_256 import *
    from .._base_.default_runtime import *
    from .._base_.models.swin_transformer_v2_base import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

# model settings
model.update(
    backbone=dict(
        arch='small',
        img_size=256,
        drop_path_rate=0.3,
        window_size=[16, 16, 16, 8]),
    head=dict(in_channels=768),
    init_cfg=[
        dict(type=TruncNormalInit, layer='Linear', std=0.02, bias=0.),
        dict(type=ConstantInit, layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(
        augments=[dict(type=Mixup, alpha=0.8),
                  dict(type=CutMix, alpha=1.0)]))

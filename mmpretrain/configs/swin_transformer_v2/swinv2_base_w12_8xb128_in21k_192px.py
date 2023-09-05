# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet21k_bs128 import *
    from .._base_.default_runtime import *
    from .._base_.models.swin_transformer_v2.base_256 import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

# model settings
model = dict(
    backbone=dict(img_size=192, window_size=[12, 12, 12, 6]),
    head=dict(num_classes=21841),
)

# dataset settings
data_preprocessor = dict(num_classes=21841)

_base_['train_pipeline'][1]['scale'] = 192  # RandomResizedCrop
_base_['test_pipeline'][1]['scale'] = 219  # ResizeEdge
_base_['test_pipeline'][2]['crop_size'] = 192  # CenterCrop

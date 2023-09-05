# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

from mmpretrain.models import ImageClassifier

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_256 import *
    from .._base_.default_runtime import *
    from .._base_.models.swin_transformer_v2.base_256 import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

model = dict(
    type=ImageClassifier,
    backbone=dict(
        window_size=[16, 16, 16, 8],
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6]))

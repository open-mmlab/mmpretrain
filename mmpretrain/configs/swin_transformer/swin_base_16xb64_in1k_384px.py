# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_384 import *
    from .._base_.default_runtime import *
    from .._base_.models.swin_transformer_base import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))

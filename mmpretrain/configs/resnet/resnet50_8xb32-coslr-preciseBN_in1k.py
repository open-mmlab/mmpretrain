# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

from mmpretrain.engine import PreciseBNHook

with read_base():
    from .._base_.datasets.imagenet_bs32 import *
    from .._base_.default_runtime import *
    from .._base_.models.resnet50 import *
    from .._base_.schedules.imagenet_bs256_coslr import *

# Precise BN hook will update the bn stats, so this hook should be executed
# before CheckpointHook(priority of 'VERY_LOW') and
# EMAHook(priority of 'NORMAL') So set the priority of PreciseBNHook to
# 'ABOVENORMAL' here.
custom_hooks = [
    dict(
        type=PreciseBNHook,
        num_samples=8192,
        interval=1,
        priority='ABOVE_NORMAL')
]

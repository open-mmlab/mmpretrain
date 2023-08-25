# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.models.mobilenet_v3.mobilenet_v3_small_cifar import *
    from .._base_.datasets.cifar10_bs16 import *
    from .._base_.schedules.cifar10_bs128 import *
    from .._base_.default_runtime import *

from mmengine.optim import MultiStepLR

# schedule settings
param_scheduler.merge(
    dict(
        type=MultiStepLR,
        by_epoch=True,
        milestones=[120, 170],
        gamma=0.1,
    ))

train_cfg.merge(dict(by_epoch=True, max_epochs=200))

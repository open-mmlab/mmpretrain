# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.cifar10_bs16 import *
    from .._base_.default_runtime import *
    from .._base_.models.resnet50_cifar_mixup import *
    from .._base_.schedules.cifar10_bs128 import *

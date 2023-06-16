# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
if '_base_':
    from .._base_.datasets.imagenet_bs32 import *
    from .._base_.default_runtime import *
    from .._base_.models.resnet18 import *
    from .._base_.schedules.imagenet_bs256 import *

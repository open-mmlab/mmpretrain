# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .group_neck import GroupNeck
from .hr_fuse import HRFuseScales
from .reduction import LinearReduction

__all__ = [
    'GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales',
    'LinearReduction', 'GroupNeck'
]

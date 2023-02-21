# Copyright (c) OpenMMLab. All rights reserved.
from .dolgnet import DolgNet
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .reduction import LinearReduction

__all__ = [
    'GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales',
    'LinearReduction', 'DolgNet'
]

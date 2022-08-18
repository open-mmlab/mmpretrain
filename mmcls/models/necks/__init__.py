# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .reduction import Reduction

__all__ = [
    'GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales',
    'Reduction'
]

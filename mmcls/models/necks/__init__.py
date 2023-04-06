# Copyright (c) OpenMMLab. All rights reserved.
from .clip_proj import CLIPProjection
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .reduction import LinearReduction

__all__ = [
    'GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales',
    'LinearReduction', 'CLIPProjection'
]

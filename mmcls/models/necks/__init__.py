# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .reduction import LinearReduction
from .transformer_token_merge import TransformerTokenMergeNeck

__all__ = [
    'GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales',
    'LinearReduction', 'TransformerTokenMergeNeck'
]

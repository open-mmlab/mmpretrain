# Copyright (c) OpenMMLab. All rights reserved.
from .augments import Augments
from .cutmix import BatchCutMixLayer
from .cutmixup import BatchCutMixupLayer
from .identity import Identity
from .mixup import BatchMixupLayer

__all__ = [
    'Augments', 'BatchCutMixLayer', 'Identity', 'BatchMixupLayer',
    'BatchCutMixupLayer'
]

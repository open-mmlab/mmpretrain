# Copyright (c) OpenMMLab. All rights reserved.
from .class_num_check_hook import ClassNumCheckHook
from .lr_updater import CosineAnnealingCooldownLrUpdaterHook
from .precise_bn_hook import PreciseBNHook
from .switch_augments_hook import SwitchDataAugHook, SwitchTrainAugHook

__all__ = [
    'ClassNumCheckHook', 'PreciseBNHook',
    'CosineAnnealingCooldownLrUpdaterHook', 'SwitchTrainAugHook',
    'SwitchDataAugHook'
]

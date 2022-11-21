# Copyright (c) OpenMMLab. All rights reserved.
from .arcface_hooks import SetAdaptiveMarginsHook
from .class_num_check_hook import ClassNumCheckHook
from .precise_bn_hook import PreciseBNHook
from .retriever_hooks import PrepareProtoBeforeValLoopHook
from .switch_recipe_hook import SwitchRecipeHook
from .visualization_hook import VisualizationHook

__all__ = [
    'ClassNumCheckHook', 'PreciseBNHook', 'VisualizationHook',
    'SwitchRecipeHook', 'PrepareProtoBeforeValLoopHook',
    'SetAdaptiveMarginsHook'
]

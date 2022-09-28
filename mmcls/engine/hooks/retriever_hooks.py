# Copyright (c) OpenMMLab. All rights reserved
from mmengine.hooks import Hook

from mmcls.registry import HOOKS


@HOOKS.register_module()
class ResetPrototypeInitFlagHook(Hook):
    """The hook that resets the prototype's initialization flag.

    During the training of the retriever, the parameters of encoder changes, so
    the `prototype_inited` needs to be set to False before validation.
    """

    def before_val(self, runner) -> None:
        if hasattr(runner.model, 'prototype'):
            runner.model.prototype_inited = False

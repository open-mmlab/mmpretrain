# Copyright (c) OpenMMLab. All rights reserved
from mmengine.hooks import Hook

from mmcls.registry import HOOKS


@HOOKS.register_module()
class PrototypeResetHook(Hook):

    def after_train(self, runner):
        if hasattr(runner.model, 'prototype'):
            runner.model.prototype_inited = False

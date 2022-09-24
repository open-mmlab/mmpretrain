# Copyright (c) OpenMMLab. All rights reserved
from mmengine.hooks import Hook

from mmcls.registry import HOOKS


@HOOKS.register_module()
class RetrieverHook(Hook):

    def after_train(self, runner):
        if hasattr(runner.model, 'prototype'):
            runner.model.prototype_inited = False

    def before_val(self, runner):
        if hasattr(runner.model, 'prototype'):
            runner.model.prepare_prototype()

    def before_test(self, runner):
        if hasattr(runner.model, 'prototype'):
            runner.model.prepare_prototype()

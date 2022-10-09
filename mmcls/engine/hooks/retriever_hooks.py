# Copyright (c) OpenMMLab. All rights reserved
import warnings

from mmengine.hooks import Hook

from mmcls.models import BaseRetriever
from mmcls.registry import HOOKS


@HOOKS.register_module()
class ResetPrototypeInitFlagHook(Hook):
    """The hook to reset the prototype's initialization flag in retrievers.

    Since the encoders of the retriever changes during training, the prototype
    changes accordingly. So the `prototype_inited` needs to be set to False
    before validation.
    """

    def before_val(self, runner) -> None:
        if isinstance(runner.model, BaseRetriever):
            if hasattr(runner.model, 'prototype_inited'):
                runner.model.prototype_inited = False
        else:
            warnings.warn(
                'Only the retriever can execute `ResetPrototypeInitFlagHook`,'
                f'but got {type(runner.model)}')

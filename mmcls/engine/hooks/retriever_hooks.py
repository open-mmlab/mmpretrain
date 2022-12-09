# Copyright (c) OpenMMLab. All rights reserved
import warnings

from mmengine.hooks import Hook

from mmcls.models import BaseRetriever
from mmcls.registry import HOOKS


@HOOKS.register_module()
class PrepareProtoBeforeValLoopHook(Hook):
    """The hook to prepare the prototype in retrievers.

    Since the encoders of the retriever changes during training, the prototype
    changes accordingly. So the `prototype_vecs` needs to be regenerated before
    validation loop.
    """

    def before_val(self, runner) -> None:
        if isinstance(runner.model, BaseRetriever):
            if hasattr(runner.model, 'prepare_prototype'):
                runner.model.prepare_prototype()
        else:
            warnings.warn(
                'Only the retrievers can execute PrepareRetrieverPrototypeHook'
                f', but got {type(runner.model)}')

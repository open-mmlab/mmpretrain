# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

from mmcls.models.builder import build_loss


@HOOKS.register_module()
class StopAugmentsHook(Hook):
    """Stop augments during training. This hook turns off the model.augments.

    A PyTorch implement of : `Data Augmentation Revisited:
    Rethinking the Distribution Gap between Clean and Augmented Data
    <https://arxiv.org/pdf/1909.09148.pdf>`_

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the model.augments.
            Default: 15.
        loss (dict): Config of classification loss after stop augments.
    """

    def __init__(self,
                 num_last_epochs=15,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0)):
        self.num_last_epochs = num_last_epochs
        self.loss = loss

    def before_train_epoch(self, runner):
        """Close augments."""
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if epoch == runner.max_epochs - self.num_last_epochs:
            runner.logger.info('Stop augments now!')
            model.augments = None
            model.head.compute_loss = build_loss(self.loss)

# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

from mmcls.models.builder import build_loss


@HOOKS.register_module()
class StopTrainAugHook(Hook):
    """Stop train aug during training. This hook turns off the model.augments.

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
            runner.logger.info('Stop train aug now!')
            model.augments = None
            model.head.compute_loss = build_loss(self.loss)


@HOOKS.register_module()
class StopDataAugHook(Hook):
    """Stop data aug during training. This hook turns off the AutoAugment and
    RandAugment.

    A PyTorch implement of : `Data Augmentation Revisited:
    Rethinking the Distribution Gap between Clean and Augmented Data
    <https://arxiv.org/pdf/1909.09148.pdf>`_

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the model.augments.
            Default: 15.
        skip_type_keys (list[str], optional): Sequence of type string to be
            skip pipeline. Default: ('AutoAugment', 'RandAugment')
    """

    def __init__(self,
                 num_last_epochs=15,
                 skip_type_keys=('AutoAugment', 'RandAugment')):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False

    def before_train_epoch(self, runner):
        """Close augments."""
        epoch = runner.epoch
        train_loader = runner.data_loader
        if epoch == runner.max_epochs - self.num_last_epochs:
            runner.logger.info('Stop data aug now!')

            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            if hasattr(train_loader.dataset, 'pipeline'):
                # for dataset
                train_loader.dataset.pipeline.update_skip_type_keys(
                    self.skip_type_keys)
            elif hasattr(train_loader.dataset.dataset, 'pipeline'):
                # for dataset wrappers
                train_loader.dataset.dataset.pipeline.update_skip_type_keys(
                    self.skip_type_keys)
            else:
                raise ValueError(
                    'train_loader.dataset or train_loader.dataset.dataset'
                    ' must have pipeline')
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True

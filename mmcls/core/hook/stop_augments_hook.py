# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

from mmcls.datasets.pipelines import Compose
from mmcls.models.builder import build_loss
from mmcls.models.utils.augment import Augments


@HOOKS.register_module()
class SwitchTrainAugHook(Hook):
    """Switch train aug during training. This hook switches the model.augments.

    A PyTorch implement of : `Data Augmentation Revisited:
    Rethinking the Distribution Gap between Clean and Augmented Data
    <https://arxiv.org/pdf/1909.09148.pdf>`_

    Args:
        action_epoch (int): switch train aug at the action_epoch.
            Default: 180.
        augments_cfg (dict, optional): the new train augments.
            Default: None.
        loss (dict): Config of classification loss after stop augments.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
    """

    def __init__(self,
                 action_epoch=180,
                 augments_cfg=None,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0)):
        self.action_epoch = action_epoch
        if augments_cfg is not None:
            self.augments = Augments(augments_cfg)
        else:
            self.augments = None
        self.loss = loss

    def before_train_epoch(self, runner):
        """Close augments."""
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if epoch + 1 == self.action_epoch:
            runner.logger.info('Switch train aug now!')
            model.augments = self.augments
            model.head.compute_loss = build_loss(self.loss)


@HOOKS.register_module()
class SwitchDataAugHook(Hook):
    """Switch data aug during training. This hook switches the data
    augmentation.

    A PyTorch implement of : `Data Augmentation Revisited:
    Rethinking the Distribution Gap between Clean and Augmented Data
    <https://arxiv.org/pdf/1909.09148.pdf>`_

    Args:
        action_epoch (int): switch data aug at the action_epoch.
            Default: 180.
        pipeline (list, optional): the new pipeline.
            a list of dict, where each element represents a operation
            defined in `mmcls.datasets.pipelines`
            Default: None.
        skip_type_keys (list[str], optional): Sequence of type string to be
            skip pipeline.
            Default: ('AutoAugment', 'RandAugment').
    """

    def __init__(self,
                 action_epoch=180,
                 pipeline=None,
                 skip_type_keys=('AutoAugment', 'RandAugment')):
        self.action_epoch = action_epoch
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False

    def before_train_epoch(self, runner):
        """Close augments."""
        epoch = runner.epoch
        train_loader = runner.data_loader
        if epoch + 1 == self.action_epoch:
            runner.logger.info('Switch data aug now!')

            if hasattr(train_loader.dataset, 'pipeline'):
                # for dataset
                if self.pipeline is not None:
                    train_loader.dataset.pipeline = self.pipeline
                train_loader.dataset.pipeline.update_skip_type_keys(
                    self.skip_type_keys)
            elif hasattr(train_loader.dataset, 'datasets'):
                # for concat dataset wrappers
                new_datasets = []
                for ds in train_loader.dataset.datasets:
                    if self.pipeline is not None:
                        ds.pipeline = self.pipeline
                    ds.pipeline.update_skip_type_keys(self.skip_type_keys)
                    new_datasets.append(ds)
                train_loader.dataset.datasets = new_datasets
            elif hasattr(train_loader.dataset.dataset, 'pipeline'):
                # for other dataset wrappers
                if self.pipeline is not None:
                    train_loader.dataset.dataset.pipeline = self.pipeline
                train_loader.dataset.dataset.pipeline.update_skip_type_keys(
                    self.skip_type_keys)
            else:
                raise ValueError(
                    'train_loader.dataset or train_loader.dataset.dataset'
                    ' must have pipeline')

            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
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

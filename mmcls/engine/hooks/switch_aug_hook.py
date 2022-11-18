# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmcv.transforms import Compose
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmcls.models.utils import RandomBatchAugment
from mmcls.registry import HOOKS, MODELS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class SwitchTrainAugHook(Hook):
    """switch configuration during the training, including data pipeline, batch
    augments and loss.

    Args:
        action_epoch (int): switch train augments at the epoch of action_epoch.
            Defaults to None.
        action_iter (int): switch train augments at the iter of action_iter.
            Defaults to None.
        pipeline (dict, str, optional): the new train pipeline.
            Defaults to 'unchange', means not changing the train pipeline.
        train_augments (dict, str, optional): the new train augments.
            Defaults to 'unchange', means not changing the train augments.
        loss (dict, str, optional): the new train loss.
            Defaults to 'unchange', means not changing the train loss.

    Example:
        >>> # in config
        >>> # deinfe new_train_pipeline, new_train_augments or new_loss
        >>> custom_hooks = [
        >>>             dict(
        >>>                 type='SwitchTrainAugHook',
        >>>                 action_epoch=37,
        >>>                 pipeline=new_train_pipeline,
        >>>                 train_augments=new_train_augments,
        >>>                 loss=new_loss),]
        >>>
        >>> # switch data augments by epoch
        >>> switch_hook = dict(
        >>>                 type='SwitchTrainAugHook',
        >>>                 pipeline=new_pipeline,
        >>>                 action_epoch=5)
        >>> runner.register_custom_hooks([switch_hook])
        >>>
        >>> # switch train augments and loss by iter
        >>> switch_hook = dict(
        >>>                 type='SwitchTrainAugHook',
        >>>                 train_augments=new_train_augments,
        >>>                 loss=new_loss,
        >>>                 action_iter=5)
        >>> runner.register_custom_hooks([switch_hook])


    Note:
        This hook would modify the ``model.data_preprocessor.batch_augments``
        , ``runner.train_loop.dataloader.dataset.pipeline`` and
        ``runner.model.head.loss`` fields.
    """
    priority = 'NORMAL'

    def __init__(self,
                 action_epoch=None,
                 action_iter=None,
                 pipeline='unchange',
                 train_augments='unchange',
                 loss='unchange'):

        if action_iter is None and action_epoch is None:
            raise ValueError('one of `action_iter` and `action_epoch` '
                             'must be set in `SwitchTrainAugHook`.')
        if action_iter is not None and action_epoch is not None:
            raise ValueError('`action_iter` and `action_epoch` should '
                             'not be both set in `SwitchTrainAugHook`.')

        if action_iter is not None:
            assert isinstance(action_iter, int) and action_iter >= 0, (
                '`action_iter` must be a number larger than 0 in '
                f'`SwitchTrainAugHook`, but got action_iter: {action_iter}')
            self.by_epoch = False
        if action_epoch is not None:
            assert isinstance(action_epoch, int) and action_epoch >= 0, (
                '`action_epoch` must be a number larger than 0 in '
                f'`SwitchTrainAugHook`, but got action_epoch: {action_epoch}')
            self.by_epoch = True

        self.action_epoch = action_epoch
        self.action_iter = action_iter

        self.pipeline = pipeline
        if pipeline != 'unchange':
            self.pipeline = Compose(pipeline)
        self._restart_dataloader = False

        self.train_augments = train_augments
        if train_augments is not None and train_augments != 'unchange':
            self.train_augments = RandomBatchAugment(**train_augments)

        self.loss = MODELS.build(loss) if loss != 'unchange' else loss

    def before_train(self, runner) -> None:
        """before run setting. If resume form a checkpoint, check whether is
        the latest processed hook, if True, do the switch process.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
        """
        # if this hook is the latest switch hook obj in the previously
        # unfinished tasks, then do the switch process before train
        if runner._resume and self._is_lastest_switch_hook(runner):
            action_milestone_str = ' after resume'
            self._do_switch(runner, action_milestone_str)

    def before_train_epoch(self, runner):
        """do before train epoch."""
        if self.by_epoch and runner.epoch + 1 == self.action_epoch:
            action_milestone_str = f' at Epoch {runner.epoch + 1}'
            self._do_switch(runner, action_milestone_str)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """do before train iter."""
        if not self.by_epoch and runner.iter + 1 == self.action_iter:
            action_milestone_str = f' at Iter {runner.iter + 1}'
            self._do_switch(runner, action_milestone_str)

    def _is_lastest_switch_hook(self, runner):
        """a helper function to judge if this hook is the latest processed
        switch hooks with the same class name in a runner."""
        # collect all the switch_hook with the same class name in a list.
        switch_hook_objs = [
            hook_obj for hook_obj in runner._hooks
            if isinstance(hook_obj, SwitchTrainAugHook)
        ]

        # get the latest swict hook based on the current iter.
        dataset_length = len(runner.train_loop.dataloader)
        cur_iter = runner.train_loop.iter
        min_gap, min_gap_idx = float('inf'), -1
        for i, switch_hook_obj in enumerate(switch_hook_objs):
            # use iter to calculate
            if switch_hook_obj.by_epoch:
                exe_iter = switch_hook_obj.action_epoch * dataset_length
            else:
                exe_iter = switch_hook_obj.action_iter

            gap = cur_iter - exe_iter
            if gap < 0:
                # this hook have not beend executed
                continue
            elif gap > 0 and min_gap > gap:
                # this hook have been executed and is closer to cur iter
                min_gap = gap
                min_gap_idx = i

        # return if self is the latest executed switch hook
        return min_gap_idx != -1 and self is switch_hook_objs[min_gap_idx]

    def _do_switch(self, runner, action_milestone_str=''):
        """do the switch aug process."""
        if self.train_augments != 'unchange':
            self._switch_batch_augments(runner)
            runner.logger.info(f'Switch train aug{action_milestone_str}.')

        if self.pipeline != 'unchange':
            self._switch_train_loader_pipeline(runner)
            runner.logger.info(f'Switch train pipeline{action_milestone_str}.')
        if self.loss != 'unchange':
            self._switch_train_loss(runner)
            runner.logger.info(f'Switch train loss{action_milestone_str}.')

    def _switch_batch_augments(self, runner):
        """switch the train augments."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        model.data_preprocessor.batch_augments = self.train_augments

    def _switch_train_loader_pipeline(self, runner):
        """switch the train loader dataset pipeline."""
        train_loader = runner.train_loop.dataloader
        if hasattr(train_loader.dataset, 'pipeline'):
            # for dataset
            if self.pipeline is not None:
                train_loader.dataset.pipeline = self.pipeline
        elif hasattr(train_loader.dataset, 'datasets'):
            # for concat dataset wrappers
            for ds in train_loader.dataset.datasets:
                ds.pipeline = self.pipeline
        elif hasattr(train_loader.dataset.dataset, 'pipeline'):
            # for other dataset wrappers
            train_loader.dataset.dataset.pipeline = self.pipeline
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

    def _switch_train_loss(self, runner):
        """switch the train loss."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        assert hasattr(model.head, 'loss')
        model.head.loss = self.loss

# Copyright (c) OpenMMLab. All rights reserved
from mmcv.runner import IterBasedRunner
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import is_seq_of


@HOOKS.register_module()
class ClassNumCheckHook(Hook):

    def _check_head(self, runner, dataset):
        """Check whether the `num_classes` in head matches the length of
        `CLASSES` in `dataset`.

        Args:
            runner (obj:`EpochBasedRunner`, `IterBasedRunner`): runner object.
            dataset (obj: `BaseDataset`): the dataset to check.
        """
        model = runner.model
        if dataset.CLASSES is None:
            runner.logger.warning(
                f'Please set `CLASSES` '
                f'in the {dataset.__class__.__name__} and'
                f'check if it is consistent with the `num_classes` '
                f'of head')
        else:
            assert is_seq_of(dataset.CLASSES, str), \
                (f'`CLASSES` in {dataset.__class__.__name__}'
                 f'should be a tuple of str.')
            for name, module in model.named_modules():
                if hasattr(module, 'num_classes'):
                    assert module.num_classes == len(dataset.CLASSES), \
                        (f'The `num_classes` ({module.num_classes}) in '
                         f'{module.__class__.__name__} of '
                         f'{model.__class__.__name__} does not matches '
                         f'the length of `CLASSES` '
                         f'{len(dataset.CLASSES)}) in '
                         f'{dataset.__class__.__name__}')

    def before_train_iter(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj: `IterBasedRunner`): Iter based Runner.
        """
        if not isinstance(runner, IterBasedRunner):
            return
        self._check_head(runner, runner.data_loader._dataloader.dataset)

    def before_val_iter(self, runner):
        """Check whether the eval dataset is compatible with head.

        Args:
            runner (obj:`IterBasedRunner`): Iter based Runner.
        """
        if not isinstance(runner, IterBasedRunner):
            return
        self._check_head(runner, runner.data_loader._dataloader.dataset)

    def before_train_epoch(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner, runner.data_loader.dataset)

    def before_val_epoch(self, runner):
        """Check whether the eval dataset is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner, runner.data_loader.dataset)

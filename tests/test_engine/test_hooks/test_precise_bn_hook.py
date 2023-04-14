# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.runner import Runner
from torch.utils.data import DataLoader, Dataset

from mmpretrain.models.utils import ClsDataPreprocessor
from mmpretrain.registry import HOOKS
from mmpretrain.structures import DataSample


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0
        self.metainfo = None

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1.0], dtype=torch.float32))
        return results

    def __len__(self):
        return 10


class MockDataPreprocessor(ClsDataPreprocessor):
    """mock preprocessor that do nothing."""

    def forward(self, data, training=False):

        return dict(inputs=data['imgs'], data_samples=DataSample())


class ExampleModel(BaseModel):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.data_preprocessor = MockDataPreprocessor()
        self.conv = nn.Linear(1, 1)
        self.bn = nn.BatchNorm1d(1)
        self.test_cfg = None

    def forward(self, inputs, data_samples, mode='tensor'):
        inputs = inputs.to(next(self.parameters()).device)
        return self.bn(self.conv(inputs))

    def train_step(self, data, optim_wrapper):
        outputs = {'loss': 0.5, 'num_samples': 1}
        return outputs


class SingleBNModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.test_cfg = None

    def forward(self, inputs, data_samples, mode='tensor'):
        return self.bn(inputs)


class GNExampleModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.gn = nn.GroupNorm(1, 1)
        self.test_cfg = None


class NoBNExampleModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        delattr(self, 'bn')
        self.test_cfg = None

    def forward(self, inputs, data_samples, mode='tensor'):
        return self.conv(inputs)


class TestPreciseBNHookHook(TestCase):
    DEFAULT_ARGS = dict(type='PreciseBNHook', num_samples=4, interval=1)
    count = 0

    def setUp(self) -> None:
        # optimizer
        self.optim_wrapper = dict(
            optimizer=dict(
                type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
        # learning policy
        self.epoch_param_scheduler = dict(
            type='MultiStepLR', by_epoch=True, milestones=[1, 2], gamma=0.1)
        self.iter_param_scheduler = dict(
            type='MultiStepLR', by_epoch=False, milestones=[1, 2], gamma=0.1)

        self.default_hooks = dict(
            timer=dict(type='IterTimerHook'),
            logger=None,
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            visualization=dict(type='VisualizationHook', enable=False),
        )
        self.epoch_train_cfg = dict(by_epoch=True, max_epochs=1)
        self.iter_train_cfg = dict(by_epoch=False, max_iters=5)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.preciseBN_cfg = copy.deepcopy(self.DEFAULT_ARGS)

        test_dataset = ExampleDataset()
        self.loader = DataLoader(test_dataset, batch_size=2)
        self.model = ExampleModel()

    def test_construct(self):
        self.runner = Runner(
            model=self.model,
            work_dir=self.tmpdir.name,
            train_dataloader=self.loader,
            train_cfg=self.epoch_train_cfg,
            log_level='WARNING',
            optim_wrapper=self.optim_wrapper,
            param_scheduler=self.epoch_param_scheduler,
            default_scope='mmpretrain',
            default_hooks=self.default_hooks,
            experiment_name='test_construct',
            custom_hooks=None)

        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        precise_bn = HOOKS.build(cfg)
        self.assertEqual(precise_bn.num_samples, 4)
        self.assertEqual(precise_bn.interval, 1)

        with pytest.raises(AssertionError):
            # num_samples must be larger than 0
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['num_samples'] = -1
            HOOKS.build(cfg)

        with pytest.raises(AssertionError):
            # interval must be larger than 0
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['interval'] = 0
            HOOKS.build(cfg)

    @patch('mmengine.dist.get_dist_info', MagicMock(return_value=(1, 2)))
    @patch('torch.distributed.all_reduce', MagicMock())
    def test_after_train_epoch_multi_machines(self):
        # Test with normal conv model in single machine
        self.preciseBN_cfg['priority'] = 'ABOVE_NORMAL'
        self.runner = Runner(
            model=self.model,
            work_dir=self.tmpdir.name,
            train_dataloader=self.loader,
            train_cfg=self.epoch_train_cfg,
            log_level='WARNING',
            optim_wrapper=self.optim_wrapper,
            param_scheduler=self.epoch_param_scheduler,
            default_scope='mmpretrain',
            default_hooks=self.default_hooks,
            experiment_name='test_after_train_epoch_multi_machines',
            custom_hooks=[self.preciseBN_cfg])
        self.runner.train()

    def test_after_train_epoch(self):
        self.preciseBN_cfg['priority'] = 'ABOVE_NORMAL'
        self.runner = Runner(
            model=self.model,
            work_dir=self.tmpdir.name,
            train_dataloader=self.loader,
            train_cfg=self.epoch_train_cfg,
            log_level='WARNING',
            optim_wrapper=self.optim_wrapper,
            param_scheduler=self.epoch_param_scheduler,
            default_scope='mmpretrain',
            default_hooks=self.default_hooks,
            experiment_name='test_after_train_epoch',
            custom_hooks=[self.preciseBN_cfg])

        # Test with normal conv model in single machine
        self.runner._train_loop = self.epoch_train_cfg
        self.runner.train()

        # Test with only BN model
        self.runner.model = SingleBNModel()
        self.runner._train_loop = self.epoch_train_cfg
        self.runner.train()

        # Test with GN model
        self.runner.model = GNExampleModel()
        self.runner._train_loop = self.epoch_train_cfg
        self.runner.train()

        # Test with no BN model
        self.runner.model = NoBNExampleModel()
        self.runner._train_loop = self.epoch_train_cfg
        self.runner.train()

    def test_after_train_iter(self):
        # test precise bn hook in iter base loop
        self.preciseBN_cfg['priority'] = 'ABOVE_NORMAL'
        test_dataset = ExampleDataset()
        self.loader = DataLoader(test_dataset, batch_size=2)
        self.runner = Runner(
            model=self.model,
            work_dir=self.tmpdir.name,
            train_dataloader=self.loader,
            train_cfg=self.iter_train_cfg,
            log_level='WARNING',
            optim_wrapper=self.optim_wrapper,
            param_scheduler=self.iter_param_scheduler,
            default_scope='mmpretrain',
            default_hooks=self.default_hooks,
            experiment_name='test_after_train_iter',
            custom_hooks=[self.preciseBN_cfg])
        self.runner.train()

    def tearDown(self) -> None:
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory.
        logging.shutdown()
        MMLogger._instance_dict.clear()
        self.tmpdir.cleanup()

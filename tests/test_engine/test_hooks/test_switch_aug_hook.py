# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import random
import tempfile
from typing import List
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from mmcv.transforms import BaseTransform
from mmengine.dataset import BaseDataset, ConcatDataset, RepeatDataset
from mmengine.logging import MMLogger
from mmengine.model import BaseDataPreprocessor, BaseModel
from mmengine.runner import Runner
from mmengine.utils import digit_version
from torch.utils.data import DataLoader

from mmcls.engine import SwitchTrainAugHook
from mmcls.models.losses import LabelSmoothLoss
from mmcls.models.utils.batch_augments import CutMix, Mixup, RandomBatchAugment
from mmcls.structures import ClsDataSample
from mmcls.utils import register_all_modules

register_all_modules()


class MockDataPreprocessor(BaseDataPreprocessor):
    """mock preprocessor that do nothing."""

    def forward(self, data, training):

        return data['imgs'], ClsDataSample()


class ExampleModel(BaseModel):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.data_preprocessor = MockDataPreprocessor()
        self.conv = nn.Linear(1, 1)
        self.bn = nn.BatchNorm1d(1)
        self.test_cfg = None

    def forward(self, batch_inputs, data_samples, mode='tensor'):
        batch_inputs = batch_inputs.to(next(self.parameters()).device)
        return self.bn(self.conv(batch_inputs))

    def train_step(self, data, optim_wrapper):
        outputs = {'loss': 0.5, 'num_samples': 1}
        return outputs


class ExampleDataset(BaseDataset):

    def load_data_list(self) -> List[dict]:
        return [i for i in range(10)]

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1.0], dtype=torch.float32))
        return results

    def __len__(self):
        return 10


class TestSwitchTrainAugHook(TestCase):
    DEFAULT_CFG = dict(action_epoch=1, action_iter=None)

    def setUp(self):
        # optimizer
        self.optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.1))
        # learning policy
        self.epochscheduler = dict(
            type='MultiStepLR', by_epoch=True, milestones=[1])
        self.iterscheduler = dict(
            type='MultiStepLR', by_epoch=False, milestones=[1])

        self.tmpdir = tempfile.TemporaryDirectory()
        self.loader = DataLoader(ExampleDataset(), batch_size=2)

    def test_init(self):
        # check action_epoch and action_iter both set
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['action_iter'] = 3
        with self.assertRaises(ValueError):
            SwitchTrainAugHook(**cfg)

        # check action_epoch and action_iter both None
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['action_epoch'] = None
        with self.assertRaises(ValueError):
            SwitchTrainAugHook(**cfg)

        # check action_epoch > 0
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['action_epoch'] = -1
        with self.assertRaises(AssertionError):
            SwitchTrainAugHook(**cfg)

        # check action_iter > 0
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['action_epoch'] = None
        cfg['action_iter'] = '3'
        with self.assertRaises(AssertionError):
            SwitchTrainAugHook(**cfg)

        # test by_epoch is True
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        hook_obj = SwitchTrainAugHook(**cfg)
        self.assertTrue(hook_obj.by_epoch)
        self.assertEqual(hook_obj.action_epoch, 1)
        self.assertEqual(hook_obj.action_iter, None)
        self.assertEqual(hook_obj.pipeline, 'unchange')
        self.assertEqual(hook_obj.train_augments, 'unchange')
        self.assertEqual(hook_obj.loss, 'unchange')

        # test by_epoch is False
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['action_epoch'] = None
        cfg['action_iter'] = 3
        hook_obj = SwitchTrainAugHook(**cfg)
        self.assertFalse(hook_obj.by_epoch)
        self.assertEqual(hook_obj.action_epoch, None)
        self.assertEqual(hook_obj.action_iter, 3)

        # test pipeline, loss
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['pipeline'] = [dict(type='LoadImageFromFile')]
        cfg['loss'] = dict(type='LabelSmoothLoss', label_smooth_val=0.1)
        hook_obj = SwitchTrainAugHook(**cfg)
        self.assertIsInstance(hook_obj.pipeline, BaseTransform)
        self.assertIsInstance(hook_obj.loss, LabelSmoothLoss)

        # test pieline is [], and train_augments
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['pipeline'] = []
        train_cfg = dict(augments=[
            dict(type='Mixup', alpha=0.8),
            dict(type='CutMix', alpha=1.0)
        ])
        cfg['train_augments'] = train_cfg
        hook_obj = SwitchTrainAugHook(**cfg)
        self.assertIsInstance(hook_obj.pipeline, BaseTransform)
        self.assertIsInstance(hook_obj.train_augments, RandomBatchAugment)

    def test_before_train_epoch(self):
        # test call once in epoch loop
        runner = self.init_runner()
        switch_hook_cfg1 = copy.deepcopy(self.DEFAULT_CFG)
        switch_hook_cfg1['type'] = 'SwitchTrainAugHook'
        runner.register_custom_hooks([switch_hook_cfg1])
        with patch.object(SwitchTrainAugHook, '_do_switch') as mock:
            runner.train()
            mock.assert_called_once()

        # test mutil call in epoch loop
        runner = self.init_runner()
        switch_hook_cfg2 = copy.deepcopy(switch_hook_cfg1)
        switch_hook_cfg3 = copy.deepcopy(switch_hook_cfg1)
        switch_hook_cfg2['action_epoch'] = 2
        switch_hook_cfg3['action_epoch'] = 3
        runner.register_custom_hooks(
            [switch_hook_cfg1, switch_hook_cfg2, switch_hook_cfg3])
        with patch.object(SwitchTrainAugHook, '_do_switch') as mock:
            runner.train()
            self.assertEqual(mock.call_count, 3)

    def test_before_train_iter(self):
        # test call once in iter loop
        runner = self.init_runner(by_epoch=False)
        switch_hook_cfg1 = copy.deepcopy(self.DEFAULT_CFG)
        switch_hook_cfg1['type'] = 'SwitchTrainAugHook'
        switch_hook_cfg1['action_epoch'] = None
        switch_hook_cfg1['action_iter'] = 1
        runner.register_custom_hooks([switch_hook_cfg1])
        with patch.object(SwitchTrainAugHook, '_do_switch') as mock:
            runner.train()
            mock.assert_called_once()

        # test mutil call in iter loop
        runner = self.init_runner(by_epoch=False)
        switch_hook_cfg2 = copy.deepcopy(switch_hook_cfg1)
        switch_hook_cfg3 = copy.deepcopy(switch_hook_cfg1)
        switch_hook_cfg2['action_iter'] = 2
        switch_hook_cfg3['action_iter'] = 3
        runner.register_custom_hooks(
            [switch_hook_cfg1, switch_hook_cfg2, switch_hook_cfg3])
        with patch.object(SwitchTrainAugHook, '_do_switch') as mock:
            runner.train()
            self.assertEqual(mock.call_count, 3)

    def test_do_switch(self):
        # test switch train augments
        runner = MagicMock()
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        train_cfg = dict(augments=[
            dict(type='Mixup', alpha=0.5),
            dict(type='CutMix', alpha=0.9)
        ])
        cfg['train_augments'] = train_cfg
        hook_obj = SwitchTrainAugHook(**cfg)
        with patch.object(SwitchTrainAugHook, '_switch_train_loss') as m_loss:
            with patch.object(SwitchTrainAugHook,
                              '_switch_train_loader_pipeline') as m_pipe:
                hook_obj._do_switch(runner)
                m_loss.assert_not_called()
                m_pipe.assert_not_called()
                runner_batchaug = runner.model.data_preprocessor.batch_augments
                self.assertIsInstance(runner_batchaug, RandomBatchAugment)
                self.assertEqual(len(runner_batchaug.augments), 2)
                self.assertIsInstance(runner_batchaug.augments[0], Mixup)
                self.assertEqual(runner_batchaug.augments[0].alpha, 0.5)
                self.assertIsInstance(runner_batchaug.augments[1], CutMix)
                self.assertEqual(runner_batchaug.augments[1].alpha, 0.9)

        # test switch data aug
        runner = MagicMock()
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['pipeline'] = [dict(type='LoadImageFromFile')]
        hook_obj = SwitchTrainAugHook(**cfg)
        with patch.object(SwitchTrainAugHook,
                          '_switch_batch_augments') as m_batch:
            with patch.object(SwitchTrainAugHook,
                              '_switch_train_loss') as m_loss:
                hook_obj._do_switch(runner)
                m_batch.assert_not_called()
                m_loss.assert_not_called()
                runner_pipeline = runner.train_loop.dataloader.dataset.pipeline
                self.assertIsInstance(runner_pipeline, BaseTransform)
                self.assertEqual(len(runner_pipeline.transforms), 1)

        # test with persistent_workers=True
        if digit_version(torch.__version__) >= digit_version('1.8.0'):
            runner = MagicMock()
            loader = DataLoader(
                ExampleDataset(), persistent_workers=True, num_workers=1)
            runner.train_loop.dataloader = loader
            cfg = copy.deepcopy(self.DEFAULT_CFG)
            cfg['pipeline'] = [
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=256)
            ]
            hook_obj = SwitchTrainAugHook(**cfg)
            hook_obj._do_switch(runner)
            runner_pipeline = runner.train_loop.dataloader.dataset.pipeline
            self.assertIsInstance(runner_pipeline, BaseTransform)
            self.assertEqual(len(runner_pipeline.transforms), 2)

        # test with ConcatDataset warpper
        runner = MagicMock()
        loader = DataLoader(
            ConcatDataset([ExampleDataset(),
                           ExampleDataset()]))
        runner.train_loop.dataloader = loader
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['pipeline'] = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=256)
        ]
        hook_obj = SwitchTrainAugHook(**cfg)
        hook_obj._do_switch(runner)
        for i in range(2):
            runner_dataset = runner.train_loop.dataloader.dataset.datasets[i]
            runner_pipeline = runner_dataset.pipeline
            self.assertIsInstance(runner_pipeline, BaseTransform)
            self.assertEqual(len(runner_pipeline.transforms), 2)

        # test with RepeatDataset warpper
        runner = MagicMock()
        loader = DataLoader(RepeatDataset(ExampleDataset(), 3))
        runner.train_loop.dataloader = loader
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['pipeline'] = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=256)
        ]
        hook_obj = SwitchTrainAugHook(**cfg)
        hook_obj._do_switch(runner)
        runner_pipeline = runner.train_loop.dataloader.dataset.dataset.pipeline
        self.assertIsInstance(runner_pipeline, BaseTransform)
        self.assertEqual(len(runner_pipeline.transforms), 2)

        # test switch loss
        runner = MagicMock()
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['loss'] = dict(type='LabelSmoothLoss', label_smooth_val=0.2)
        hook_obj = SwitchTrainAugHook(**cfg)
        with patch.object(SwitchTrainAugHook,
                          '_switch_batch_augments') as m_batch:
            with patch.object(SwitchTrainAugHook,
                              '_switch_train_loader_pipeline') as m_pipe:
                hook_obj._do_switch(runner)
                m_batch.assert_not_called()
                m_pipe.assert_not_called()
                runner_loss = runner.model.head.loss
                self.assertIsInstance(runner_loss, nn.Module)
                self.assertTrue(runner_loss.label_smooth_val, 0.2)

        # test both
        runner = MagicMock()
        cfg = copy.deepcopy(self.DEFAULT_CFG)
        cfg['pipeline'] = [dict(type='LoadImageFromFile')]
        cfg['loss'] = dict(type='LabelSmoothLoss', label_smooth_val=0.2)
        cfg['train_augments'] = dict(augments=[dict(type='Mixup', alpha=0.5)])
        hook_obj = SwitchTrainAugHook(**cfg)
        with patch.object(SwitchTrainAugHook,
                          '_switch_batch_augments') as m_batch:
            with patch.object(SwitchTrainAugHook,
                              '_switch_train_loader_pipeline') as m_pipe:
                with patch.object(SwitchTrainAugHook,
                                  '_switch_train_loss') as m_loss:
                    hook_obj._do_switch(runner)
                    m_batch.assert_called_once()
                    m_pipe.assert_called_once()
                    m_loss.assert_called_once()

    def create_patch(self, object, name):
        patcher = patch.object(object, name)
        thing = patcher.start()
        self.addCleanup(patcher.stop)
        return thing

    def test_before_train(self):
        # test not resume
        runner = self.init_runner(resume=False, epoch=2)
        hook_obj1 = SwitchTrainAugHook(action_epoch=4)
        hook_obj2 = SwitchTrainAugHook(action_epoch=7)
        runner.register_custom_hooks([hook_obj1, hook_obj2])
        mock_hook1 = self.create_patch(hook_obj1, '_do_switch')
        mock_hook2 = self.create_patch(hook_obj2, '_do_switch')
        runner.call_hook('before_train')
        mock_hook1.assert_not_called()
        mock_hook2.assert_not_called()

        # test resume from no processed switch hook
        runner = self.init_runner(resume=True, epoch=2)
        hook_obj1 = SwitchTrainAugHook(action_epoch=4)
        hook_obj2 = SwitchTrainAugHook(action_epoch=7)
        runner.register_custom_hooks([hook_obj1, hook_obj2])
        mock_hook1 = self.create_patch(hook_obj1, '_do_switch')
        mock_hook2 = self.create_patch(hook_obj2, '_do_switch')
        runner.call_hook('before_train')
        mock_hook1.assert_not_called()
        mock_hook2.assert_not_called()

        # test resume from epoch processed switch hook
        runner = self.init_runner(resume=True, epoch=5)
        hook_obj1 = SwitchTrainAugHook(action_epoch=2)
        hook_obj2 = SwitchTrainAugHook(action_epoch=7)
        hook_obj3 = SwitchTrainAugHook(action_epoch=3)
        runner.register_custom_hooks([hook_obj1, hook_obj2, hook_obj3])
        mock_hook1 = self.create_patch(hook_obj1, '_do_switch')
        mock_hook2 = self.create_patch(hook_obj2, '_do_switch')
        mock_hook3 = self.create_patch(hook_obj3, '_do_switch')
        runner.call_hook('before_train')
        mock_hook1.assert_not_called()
        mock_hook2.assert_not_called()
        mock_hook3.assert_called_once()

        # test resume from iter processed switch hook
        runner = self.init_runner(resume=True, iter=15, by_epoch=False)
        hook_obj1 = SwitchTrainAugHook(action_iter=2)
        hook_obj2 = SwitchTrainAugHook(action_iter=12)
        hook_obj3 = SwitchTrainAugHook(action_epoch=18)
        runner.register_custom_hooks([hook_obj1, hook_obj2, hook_obj3])
        mock_hook1 = self.create_patch(hook_obj1, '_do_switch')
        mock_hook2 = self.create_patch(hook_obj2, '_do_switch')
        mock_hook3 = self.create_patch(hook_obj3, '_do_switch')
        runner.call_hook('before_train')
        mock_hook1.assert_not_called()
        mock_hook2.assert_called_once()
        mock_hook3.assert_not_called()

        # test resume from epoch loop and iter hook
        runner = self.init_runner(resume=True, epoch=1)  # i epoch = 5 iter
        hook_obj1 = SwitchTrainAugHook(action_iter=2)
        hook_obj2 = SwitchTrainAugHook(action_iter=15)
        hook_obj3 = SwitchTrainAugHook(action_iter=7)
        runner.register_custom_hooks([hook_obj1, hook_obj2, hook_obj3])
        mock_hook1 = self.create_patch(hook_obj1, '_do_switch')
        mock_hook2 = self.create_patch(hook_obj2, '_do_switch')
        mock_hook3 = self.create_patch(hook_obj3, '_do_switch')
        runner.call_hook('before_train')
        mock_hook1.assert_called_once()
        mock_hook2.assert_not_called()
        mock_hook3.assert_not_called()

    def init_runner(self, resume=False, epoch=None, iter=None, by_epoch=True):
        if by_epoch:
            runner = Runner(
                model=ExampleModel(),
                work_dir=self.tmpdir.name,
                train_dataloader=self.loader,
                optim_wrapper=self.optim_wrapper,
                param_scheduler=self.epochscheduler,
                train_cfg=dict(by_epoch=True, max_epochs=3),
                default_hooks=dict(logger=None),
                log_processor=dict(window_size=1),
                experiment_name=f'test_{resume}_{random.random()}',
                default_scope='mmcls')
        else:
            runner = Runner(
                model=ExampleModel(),
                work_dir=self.tmpdir.name,
                train_dataloader=self.loader,
                optim_wrapper=self.optim_wrapper,
                param_scheduler=self.iterscheduler,
                train_cfg=dict(by_epoch=False, max_iters=3),
                default_hooks=dict(logger=None),
                log_processor=dict(window_size=1),
                experiment_name=f'test_{resume}_{random.random()}',
                default_scope='mmcls')
        runner._resume = resume
        dataset_length = len(self.loader)
        if epoch and by_epoch:
            runner.train_loop._epoch = epoch
            runner.train_loop._iter = epoch * dataset_length
        if iter and not by_epoch:
            runner.train_loop._iter = iter
        return runner

    def tearDown(self) -> None:
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory.
        logging.shutdown()
        MMLogger._instance_dict.clear()
        self.tmpdir.cleanup()

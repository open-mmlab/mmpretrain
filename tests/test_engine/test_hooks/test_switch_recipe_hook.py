# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import tempfile
from typing import List
from unittest import TestCase
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from mmcv.transforms import Compose
from mmengine.dataset import BaseDataset, ConcatDataset, RepeatDataset
from mmengine.device import get_device
from mmengine.logging import MMLogger
from mmengine.model import BaseDataPreprocessor, BaseModel
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner

from mmpretrain.engine import SwitchRecipeHook
from mmpretrain.models import CrossEntropyLoss
from mmpretrain.models.heads.cls_head import ClsHead
from mmpretrain.models.losses import LabelSmoothLoss
from mmpretrain.models.utils.batch_augments import RandomBatchAugment


class SimpleDataPreprocessor(BaseDataPreprocessor):

    def __init__(self):
        super().__init__()
        self.batch_augments = None

    def forward(self, data, training):

        data = self.cast_data(data)
        assert 'inputs' in data, 'No `input` in data.'
        inputs = data['inputs']
        labels = data['labels']

        if self.batch_augments is not None and training:
            inputs, labels = self.batch_augments(inputs, labels)

        return {'inputs': inputs, 'labels': labels}


class SimpleModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.data_preprocessor = SimpleDataPreprocessor()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 10)
        self.loss_module = CrossEntropyLoss(use_soft=True)

    def forward(self, inputs, labels, mode='tensor'):
        if mode == 'loss':
            score = self.fc(self.gap(inputs).view(inputs.size(0), -1))
            loss = self.loss_module(score, labels)
            return {'loss': loss}
        else:
            return None


class ExampleDataset(BaseDataset):

    def load_data_list(self) -> List[dict]:
        return [{
            'inputs': torch.rand(3, 12, 12),
            'labels': torch.rand(10),
        } for _ in range(10)]


class EmptyTransform:

    def __call__(self, results):
        return {}


class TestSwitchRecipeHook(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        logging.shutdown()
        MMLogger._instance_dict.clear()
        self.tmpdir.cleanup()

    def test_init(self):
        # test `action_epoch` is set
        with self.assertRaisesRegex(AssertionError, 'Please set'):
            SwitchRecipeHook([dict(batch_augments=None)])

        # test `action_epoch` is not repeated
        with self.assertRaisesRegex(AssertionError, 'is repeated'):
            SwitchRecipeHook([dict(action_epoch=1), dict(action_epoch=1)])

        # test recipe build
        hook = SwitchRecipeHook([
            dict(
                action_epoch=1,
                train_pipeline=[dict(type='LoadImageFromFile')],
                loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
                batch_augments=dict(augments=dict(type='Mixup', alpha=0.8)),
            )
        ])
        self.assertIn(1, hook.schedule)
        self.assertIsInstance(hook.schedule[1]['train_pipeline'], Compose)
        self.assertIsInstance(hook.schedule[1]['loss'], LabelSmoothLoss)
        self.assertIsInstance(hook.schedule[1]['batch_augments'],
                              RandomBatchAugment)

        # test recipe build with instance
        hook = SwitchRecipeHook([
            dict(
                action_epoch=1,
                train_pipeline=[MagicMock()],
                loss=MagicMock(),
                batch_augments=MagicMock(),
            )
        ])
        self.assertIn(1, hook.schedule)
        self.assertIsInstance(hook.schedule[1]['train_pipeline'], Compose)
        self.assertIsInstance(hook.schedule[1]['loss'], MagicMock)
        self.assertIsInstance(hook.schedule[1]['batch_augments'], MagicMock)

        # test empty pieline and train_augments
        hook = SwitchRecipeHook(
            [dict(action_epoch=1, train_pipeline=[], batch_augments=None)])
        self.assertIn(1, hook.schedule)
        self.assertIsInstance(hook.schedule[1]['train_pipeline'], Compose)
        self.assertIsNone(hook.schedule[1]['batch_augments'])

    def test_do_switch(self):
        device = get_device()
        model = SimpleModel().to(device)

        loss = CrossEntropyLoss(use_soft=True)
        loss.forward = MagicMock(
            side_effect=lambda x, y: CrossEntropyLoss.forward(loss, x, y))
        batch_augments = RandomBatchAugment(dict(type='Mixup', alpha=0.5))
        switch_hook = SwitchRecipeHook([
            dict(
                action_epoch=2,
                train_pipeline=[MagicMock(side_effect=lambda x: x)],
                loss=loss,
                batch_augments=MagicMock(
                    side_effect=lambda x, y: batch_augments(x, y)),
            )
        ])

        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=ExampleDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=5,
                num_workers=0,
                collate_fn=dict(type='default_collate'),
            ),
            optim_wrapper=OptimWrapper(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.)),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=10),
            work_dir=self.tmpdir.name,
            default_hooks=dict(logger=None),
            custom_hooks=[switch_hook],
            default_scope='mmpretrain',
            experiment_name='test_switch')
        runner.train()
        self.assertEqual(switch_hook.schedule[2]['batch_augments'].call_count,
                         2)
        self.assertEqual(switch_hook.schedule[2]['loss'].forward.call_count, 2)
        self.assertEqual(
            switch_hook.schedule[2]['train_pipeline'].transforms[0].call_count,
            10)

        # Due to the unknown error in Windows environment, the unit test for
        # `num_workers>0` is disabled temporarily

        # switch_hook = SwitchRecipeHook(
        #     [dict(
        #         action_epoch=2,
        #         train_pipeline=[EmptyTransform()],
        #     )])

        # runner = Runner(
        #     model=model,
        #     train_dataloader=dict(
        #         dataset=ExampleDataset(),
        #         sampler=dict(type='DefaultSampler', shuffle=True),
        #         batch_size=5,
        #         num_workers=1,
        #         persistent_workers=True,
        #         collate_fn=dict(type='default_collate'),
        #     ),
        #     optim_wrapper=OptimWrapper(
        #         optimizer=torch.optim.Adam(model.parameters(), lr=0.)),
        #     train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=10),
        #     work_dir=self.tmpdir.name,
        #     default_hooks=dict(logger=None),
        #     custom_hooks=[switch_hook],
        #     default_scope='mmpretrain',
        #     experiment_name='test_switch_multi_workers')
        # with self.assertRaisesRegex(AssertionError, 'No `input` in data.'):
        #     # If the pipeline switch works, the data_preprocessor cannot
        #     # receive `inputs`.
        #     runner.train()

    def test_resume(self):
        device = get_device()
        model = SimpleModel().to(device)

        loss = CrossEntropyLoss(use_soft=True)
        loss.forward = MagicMock(
            side_effect=lambda x, y: CrossEntropyLoss.forward(loss, x, y))
        batch_augments = RandomBatchAugment(dict(type='Mixup', alpha=0.5))
        switch_hook = SwitchRecipeHook([
            dict(
                action_epoch=1,
                train_pipeline=[MagicMock(side_effect=lambda x: x)]),
            dict(action_epoch=2, loss=loss),
            dict(
                action_epoch=4,
                batch_augments=MagicMock(
                    side_effect=lambda x, y: batch_augments(x, y)),
            ),
        ])

        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=ExampleDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=5,
                num_workers=0,
                collate_fn=dict(type='default_collate'),
            ),
            optim_wrapper=OptimWrapper(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.)),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=10),
            work_dir=self.tmpdir.name,
            default_hooks=dict(logger=None),
            custom_hooks=[switch_hook],
            default_scope='mmpretrain',
            experiment_name='test_resume1')
        runner.train()

        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=ExampleDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=5,
                num_workers=0,
                collate_fn=dict(type='default_collate'),
            ),
            optim_wrapper=OptimWrapper(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.)),
            train_cfg=dict(by_epoch=True, max_epochs=4, val_interval=10),
            resume=True,
            load_from=osp.join(self.tmpdir.name, 'epoch_2.pth'),
            work_dir=self.tmpdir.name,
            default_hooks=dict(logger=None),
            custom_hooks=[switch_hook],
            default_scope='mmpretrain',
            experiment_name='test_resume2')

        with self.assertLogs(runner.logger, 'INFO') as logs:
            runner.train()
        prefix = 'INFO:mmengine:'
        self.assertIn(
            prefix + 'Switch train pipeline (resume recipe of epoch 1).',
            logs.output)
        self.assertIn(prefix + 'Switch loss (resume recipe of epoch 2).',
                      logs.output)
        self.assertIn(prefix + 'Switch batch augments at epoch 4.',
                      logs.output)

    def test_switch_train_pipeline(self):
        device = get_device()
        model = SimpleModel().to(device)

        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=ConcatDataset([ExampleDataset(),
                                       ExampleDataset()]),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=5,
                num_workers=0,
                collate_fn=dict(type='default_collate'),
            ),
            optim_wrapper=OptimWrapper(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.)),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=10),
            work_dir=self.tmpdir.name,
            default_hooks=dict(logger=None),
            default_scope='mmpretrain',
            experiment_name='test_concat_dataset')
        pipeline = MagicMock()
        SwitchRecipeHook._switch_train_pipeline(runner, pipeline)
        self.assertIs(runner.train_dataloader.dataset.datasets[0].pipeline,
                      pipeline)
        self.assertIs(runner.train_dataloader.dataset.datasets[1].pipeline,
                      pipeline)

        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=RepeatDataset(ExampleDataset(), 3),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=5,
                num_workers=0,
                collate_fn=dict(type='default_collate'),
            ),
            optim_wrapper=OptimWrapper(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.)),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=10),
            work_dir=self.tmpdir.name,
            default_hooks=dict(logger=None),
            default_scope='mmpretrain',
            experiment_name='test_repeat_dataset')
        pipeline = MagicMock()
        SwitchRecipeHook._switch_train_pipeline(runner, pipeline)
        self.assertIs(runner.train_dataloader.dataset.dataset.pipeline,
                      pipeline)

    def test_switch_loss(self):
        device = get_device()
        model = SimpleModel().to(device)

        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=ExampleDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=5,
                num_workers=0,
                collate_fn=dict(type='default_collate'),
            ),
            optim_wrapper=OptimWrapper(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.)),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=10),
            work_dir=self.tmpdir.name,
            default_hooks=dict(logger=None),
            default_scope='mmpretrain',
            experiment_name='test_model_loss')
        loss = CrossEntropyLoss(use_soft=True)
        SwitchRecipeHook._switch_loss(runner, loss)
        self.assertIs(runner.model.loss_module, loss)

        model.add_module('head', ClsHead())
        del model.loss_module
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=ExampleDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=5,
                num_workers=0,
                collate_fn=dict(type='default_collate'),
            ),
            optim_wrapper=OptimWrapper(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.)),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=10),
            work_dir=self.tmpdir.name,
            default_hooks=dict(logger=None),
            default_scope='mmpretrain',
            experiment_name='test_head_loss')
        loss = CrossEntropyLoss(use_soft=True)
        SwitchRecipeHook._switch_loss(runner, loss)
        self.assertIs(runner.model.head.loss_module, loss)

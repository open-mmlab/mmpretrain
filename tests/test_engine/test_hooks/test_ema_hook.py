# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import tempfile
from collections import OrderedDict
from unittest import TestCase
from unittest.mock import ANY, MagicMock, call

import torch
import torch.nn as nn
from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
from mmengine.testing import assert_allclose
from torch.utils.data import Dataset

from mmpretrain.engine import EMAHook


class SimpleModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.para = nn.Parameter(torch.zeros(1))

    def forward(self, *args, mode='tensor', **kwargs):
        if mode == 'predict':
            return self.para.clone()
        elif mode == 'loss':
            return {'loss': self.para.mean()}


class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(6, 2)
    label = torch.ones(6)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class TestEMAHook(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        state_dict = OrderedDict(
            meta=dict(epoch=1, iter=2),
            # The actual ema para
            state_dict={'para': torch.tensor([1.])},
            # The actual original para
            ema_state_dict={'module.para': torch.tensor([2.])},
        )
        self.ckpt = osp.join(self.temp_dir.name, 'ema.pth')
        torch.save(state_dict, self.ckpt)

    def tearDown(self):
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()
        self.temp_dir.cleanup()

    def test_load_state_dict(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = SimpleModel().to(device)
        ema_hook = EMAHook()
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            optim_wrapper=OptimWrapper(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.)),
            train_cfg=dict(by_epoch=True, max_epochs=2),
            work_dir=self.temp_dir.name,
            resume=False,
            load_from=self.ckpt,
            default_hooks=dict(logger=None),
            custom_hooks=[ema_hook],
            default_scope='mmpretrain',
            experiment_name='load_state_dict')
        runner.train()
        assert_allclose(runner.model.para, torch.tensor([1.], device=device))

    def test_evaluate_on_ema(self):

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = SimpleModel().to(device)

        # Test validate on ema model
        evaluator = Evaluator([MagicMock()])
        runner = Runner(
            model=model,
            val_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=evaluator,
            val_cfg=dict(),
            work_dir=self.temp_dir.name,
            load_from=self.ckpt,
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook')],
            default_scope='mmpretrain',
            experiment_name='validate_on_ema')
        runner.val()
        evaluator.metrics[0].process.assert_has_calls([
            call(ANY, [torch.tensor([1.]).to(device)]),
        ])
        self.assertNotIn(
            call(ANY, [torch.tensor([2.]).to(device)]),
            evaluator.metrics[0].process.mock_calls)

        # Test test on ema model
        evaluator = Evaluator([MagicMock()])
        runner = Runner(
            model=model,
            test_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            test_evaluator=evaluator,
            test_cfg=dict(),
            work_dir=self.temp_dir.name,
            load_from=self.ckpt,
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook')],
            default_scope='mmpretrain',
            experiment_name='test_on_ema')
        runner.test()
        evaluator.metrics[0].process.assert_has_calls([
            call(ANY, [torch.tensor([1.]).to(device)]),
        ])
        self.assertNotIn(
            call(ANY, [torch.tensor([2.]).to(device)]),
            evaluator.metrics[0].process.mock_calls)

        # Test validate on both models
        evaluator = Evaluator([MagicMock()])
        runner = Runner(
            model=model,
            val_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_evaluator=evaluator,
            val_cfg=dict(),
            work_dir=self.temp_dir.name,
            load_from=self.ckpt,
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook', evaluate_on_origin=True)],
            default_scope='mmpretrain',
            experiment_name='validate_on_ema_false',
        )
        runner.val()
        evaluator.metrics[0].process.assert_has_calls([
            call(ANY, [torch.tensor([1.]).to(device)]),
            call(ANY, [torch.tensor([2.]).to(device)]),
        ])

        # Test test on both models
        evaluator = Evaluator([MagicMock()])
        runner = Runner(
            model=model,
            test_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            test_evaluator=evaluator,
            test_cfg=dict(),
            work_dir=self.temp_dir.name,
            load_from=self.ckpt,
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook', evaluate_on_origin=True)],
            default_scope='mmpretrain',
            experiment_name='test_on_ema_false',
        )
        runner.test()
        evaluator.metrics[0].process.assert_has_calls([
            call(ANY, [torch.tensor([1.]).to(device)]),
            call(ANY, [torch.tensor([2.]).to(device)]),
        ])

        # Test evaluate_on_ema=False
        evaluator = Evaluator([MagicMock()])
        with self.assertWarnsRegex(UserWarning, 'evaluate_on_origin'):
            runner = Runner(
                model=model,
                test_dataloader=dict(
                    dataset=DummyDataset(),
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    batch_size=3,
                    num_workers=0),
                test_evaluator=evaluator,
                test_cfg=dict(),
                work_dir=self.temp_dir.name,
                load_from=self.ckpt,
                default_hooks=dict(logger=None),
                custom_hooks=[dict(type='EMAHook', evaluate_on_ema=False)],
                default_scope='mmpretrain',
                experiment_name='not_test_on_ema')
        runner.test()
        evaluator.metrics[0].process.assert_has_calls([
            call(ANY, [torch.tensor([2.]).to(device)]),
        ])
        self.assertNotIn(
            call(ANY, [torch.tensor([1.]).to(device)]),
            evaluator.metrics[0].process.mock_calls)

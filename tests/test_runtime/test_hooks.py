# Copyright (c) OpenMMLab. All rights reserved.
import logging
import shutil
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.runner import build_runner
from mmcv.runner.hooks import Hook, IterTimerHook
from torch.utils.data import DataLoader, Dataset

import mmcls.core  # noqa: F401
from mmcls.datasets.pipelines import Compose
from mmcls.models.losses import CrossEntropyLoss


def _build_demo_runner_without_hook(runner_type='EpochBasedRunner',
                                    max_epochs=1,
                                    max_iters=None,
                                    multi_optimziers=False):

    class Head:

        def __init__(self):
            self.compute_loss = None

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.conv = nn.Conv2d(3, 3, 3)
            self.augments = True  # demo
            self.head = Head()

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()

    if multi_optimziers:
        optimizer = {
            'model1':
            torch.optim.SGD(model.linear.parameters(), lr=0.02, momentum=0.95),
            'model2':
            torch.optim.SGD(model.conv.parameters(), lr=0.01, momentum=0.9),
        }
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    tmp_dir = tempfile.mkdtemp()
    runner = build_runner(
        dict(type=runner_type),
        default_args=dict(
            model=model,
            work_dir=tmp_dir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_epochs=max_epochs,
            max_iters=max_iters))
    return runner


def _build_demo_runner(runner_type='EpochBasedRunner',
                       max_epochs=1,
                       max_iters=None,
                       multi_optimziers=False):

    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    runner = _build_demo_runner_without_hook(runner_type, max_epochs,
                                             max_iters, multi_optimziers)

    runner.register_checkpoint_hook(dict(interval=1))
    runner.register_logger_hooks(log_config)
    return runner


class ValueCheckHook(Hook):

    def __init__(self, check_dict, by_epoch=False):
        super().__init__()
        self.check_dict = check_dict
        self.by_epoch = by_epoch

    def after_iter(self, runner):
        if self.by_epoch:
            return
        if runner.iter in self.check_dict:
            for attr, target in self.check_dict[runner.iter].items():
                value = eval(f'runner.{attr}')
                assert np.isclose(value, target), \
                    (f'The value of `runner.{attr}` is {value}, '
                     f'not equals to {target}')

    def after_epoch(self, runner):
        if not self.by_epoch:
            return
        if runner.epoch in self.check_dict:
            for attr, target in self.check_dict[runner.epoch]:
                value = eval(f'runner.{attr}')
                assert np.isclose(value, target), \
                    (f'The value of `runner.{attr}` is {value}, '
                     f'not equals to {target}')


class ExampleAug:

    def __call__(self, results):
        return results


class ExampleDataset(Dataset):

    def __init__(self):
        self.pipeline = Compose([ExampleAug()])

    def __getitem__(self, idx):
        results = dict(x=torch.tensor([1., 1.]))
        return self.pipeline(results)['x']

    def __len__(self):
        return 10


@pytest.mark.parametrize('multi_optimziers', (True, False))
def test_cosine_cooldown_hook(multi_optimziers):
    """xdoctest -m tests/test_hooks.py test_cosine_runner_hook."""
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

    # add momentum LR scheduler
    hook_cfg = dict(
        type='CosineAnnealingCooldownLrUpdaterHook',
        by_epoch=False,
        cool_down_time=2,
        cool_down_ratio=0.1,
        min_lr_ratio=0.1,
        warmup_iters=2,
        warmup_ratio=0.9)
    runner.register_hook_from_cfg(hook_cfg)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))
    runner.register_hook(IterTimerHook())

    if multi_optimziers:
        check_hook = ValueCheckHook({
            0: {
                'current_lr()["model1"][0]': 0.02,
                'current_lr()["model2"][0]': 0.01,
            },
            5: {
                'current_lr()["model1"][0]': 0.0075558491,
                'current_lr()["model2"][0]': 0.0037779246,
            },
            9: {
                'current_lr()["model1"][0]': 0.0002,
                'current_lr()["model2"][0]': 0.0001,
            }
        })
    else:
        check_hook = ValueCheckHook({
            0: {
                'current_lr()[0]': 0.02,
            },
            5: {
                'current_lr()[0]': 0.0075558491,
            },
            9: {
                'current_lr()[0]': 0.0002,
            }
        })
    runner.register_hook(check_hook, priority='LOWEST')

    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)


def test_stop_augments_hook():
    loader = DataLoader(ExampleDataset())
    runner = _build_demo_runner(max_epochs=3)

    # add momentum LR scheduler
    hook_cfg = dict(
        type='StopTrainAugHook',
        num_last_epochs=1,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0))
    runner.register_hook_from_cfg(hook_cfg)

    hook_cfg = dict(
        type='StopDataAugHook',
        num_last_epochs=1,
        skip_type_keys=('ExampleAug'))
    runner.register_hook_from_cfg(hook_cfg)

    assert runner.model.augments
    assert runner.model.head.compute_loss is None
    assert loader.dataset.pipeline.skip_type_keys is None
    runner.run([loader], [('train', 1)])
    assert runner.model.augments is None
    assert isinstance(runner.model.head.compute_loss, CrossEntropyLoss)
    assert runner.data_loader.dataset.pipeline.skip_type_keys == \
           hook_cfg['skip_type_keys']
    shutil.rmtree(runner.work_dir)

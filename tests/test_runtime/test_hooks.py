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
from torch.utils.data import DataLoader

import mmcls.core  # noqa: F401


def _build_demo_runner_without_hook(runner_type='EpochBasedRunner',
                                    max_epochs=1,
                                    max_iters=None,
                                    multi_optimziers=False):

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.conv = nn.Conv2d(3, 3, 3)

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

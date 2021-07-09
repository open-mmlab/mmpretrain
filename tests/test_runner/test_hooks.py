import logging
import shutil
import tempfile

import pytest
import torch
import torch.nn as nn
from mmcv.runner import RUNNERS
from mmcv.runner import Fp16OptimizerHook as _Fp16OptimizerHook
from mmcv.runner import OptimizerHook as _OptimizerHook
from mmcv.runner import auto_fp16
from torch.utils.data import DataLoader

from mmcls.runner import Fp16OptimizerHook, OptimizerHook


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.fp16_enabled = False
        self.fc = nn.Linear(3, 2)
        nn.init.constant_(self.fc.weight, 1.)
        nn.init.constant_(self.fc.bias, 1.)

    @auto_fp16(apply_to=('x', ))
    def forward(self, x):
        return self.fc(x)

    def train_step(self, x, optimizer, **kwargs):
        return dict(loss=self(x).mean(), num_samples=x.shape[0])

    def val_step(self, x, optimizer, **kwargs):
        return dict(loss=self(x).mean(), num_samples=x.shape[0])


def build_toy_runner(config=dict()):
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    tmp_dir = tempfile.mkdtemp()

    if 'type' not in config:
        config['type'] = 'EpochBasedRunner'

    runner = RUNNERS.build(
        config,
        default_args=dict(
            model=model,
            work_dir=tmp_dir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_epochs=1,
            meta=dict()))
    return runner


def test_optimizer_hook():
    # test accumulation_step

    # test assertion
    with pytest.raises(AssertionError):
        # accumulation_step only accepts int
        OptimizerHook(accumulation_step='str')

    with pytest.raises(AssertionError):
        # accumulation_step only accepts positive number
        OptimizerHook(accumulation_step=-1)

    # optimize with accumulation_step
    loader_1 = DataLoader(torch.ones((6, 3)), batch_size=1)
    runner_1 = build_toy_runner()
    runner_1.register_hook(
        OptimizerHook(grad_clip=dict(max_norm=0.2), accumulation_step=3))
    runner_1.run([loader_1], [('train', 1)])

    # optimize without accumulation_step
    loader_2 = DataLoader(torch.ones((6, 3)), batch_size=3)
    runner_2 = build_toy_runner()
    runner_2.register_hook(_OptimizerHook(grad_clip=dict(max_norm=0.2)))
    runner_2.run([loader_2], [('train', 1)])

    # test optimizer works well
    assert (runner_1.model.fc.weight < 1).all()
    assert (runner_1.model.fc.bias < 1).all()
    # test optimizer with accumulation_step gets the same results
    assert torch.allclose(runner_1.model.fc.weight, runner_2.model.fc.weight)
    assert torch.allclose(runner_1.model.fc.bias, runner_2.model.fc.bias)
    shutil.rmtree(runner_1.work_dir)
    shutil.rmtree(runner_2.work_dir)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_fp16_optimzier_hook():
    # test accumulation_step

    # test assertion
    with pytest.raises(AssertionError):
        # accumulation_step only accepts int
        OptimizerHook(accumulation_step='str')

    with pytest.raises(AssertionError):
        # accumulation_step only accepts positive number
        OptimizerHook(accumulation_step=-1)

    # optimize with accumulation_step
    loader_1 = DataLoader(torch.ones((6, 3)).cuda(), batch_size=1)
    runner_1 = build_toy_runner()
    runner_1.model.cuda()
    runner_1.register_hook(
        Fp16OptimizerHook(grad_clip=dict(max_norm=0.2), accumulation_step=3))
    runner_1.run([loader_1], [('train', 1)])

    # optimize without accumulation_step
    loader_2 = DataLoader(torch.ones((6, 3)).cuda(), batch_size=3)
    runner_2 = build_toy_runner()
    runner_2.model.cuda()
    runner_2.register_hook(_Fp16OptimizerHook(grad_clip=dict(max_norm=0.2)))
    runner_2.run([loader_2], [('train', 1)])

    # test optimizer works well
    assert (runner_1.model.fc.weight < 1).all()
    assert (runner_1.model.fc.bias < 1).all()
    # test optimizer with accumulation_step gets the same results
    assert torch.allclose(runner_1.model.fc.weight, runner_2.model.fc.weight)
    assert torch.allclose(runner_1.model.fc.bias, runner_2.model.fc.bias)
    shutil.rmtree(runner_1.work_dir)
    shutil.rmtree(runner_2.work_dir)

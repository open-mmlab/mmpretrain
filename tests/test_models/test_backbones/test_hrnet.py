# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import HRNet


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


@pytest.mark.parametrize('base_channels', [18, 30, 32, 40, 44, 48, 64])
def test_hrnet_arch_zoo(base_channels):

    cfg_ori = dict(arch=f'w{base_channels}')

    # Test HRNet model with input size of 224
    model = HRNet(**cfg_ori)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(3, 3, 224, 224)
    outs = model(imgs)
    out_channels = base_channels
    out_size = 56
    assert isinstance(outs, tuple)
    for out in outs:
        assert out.shape == (3, out_channels, out_size, out_size)
        out_channels = out_channels * 2
        out_size = out_size // 2


def test_hrnet_custom_arch():

    cfg_ori = dict(
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BOTTLENECK',
                num_blocks=(4, 4, 2),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 3, 4, 4),
                num_channels=(32, 64, 152, 256)),
        ), )

    # Test HRNet model with input size of 224
    model = HRNet(**cfg_ori)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(3, 3, 224, 224)
    outs = model(imgs)
    out_channels = (32, 64, 152, 256)
    out_size = 56
    assert isinstance(outs, tuple)
    for out, out_channel in zip(outs, out_channels):
        assert out.shape == (3, out_channel, out_size, out_size)
        out_size = out_size // 2

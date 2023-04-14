# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile

import pytest
import torch
from mmengine.runner import load_checkpoint, save_checkpoint
from torch import nn
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import RepLKNet
from mmpretrain.models.backbones.replknet import ReparamLargeKernelConv


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def is_replk_block(modules):
    if isinstance(modules, ReparamLargeKernelConv):
        return True
    return False


def test_replknet_replkblock():
    # Test ReparamLargeKernelConv with in_channels != out_channels,
    # kernel_size = 31, stride = 1, groups=in_channels, small_kernel = 5
    block = ReparamLargeKernelConv(
        5, 10, kernel_size=31, stride=1, groups=5, small_kernel=5)
    block.eval()
    x = torch.randn(1, 5, 64, 64)
    x_out_not_deploy = block(x)
    assert block.small_kernel <= block.kernel_size
    assert not hasattr(block, 'lkb_reparam')
    assert hasattr(block, 'lkb_origin')
    assert hasattr(block, 'small_conv')
    assert x_out_not_deploy.shape == torch.Size((1, 10, 64, 64))
    block.merge_kernel()
    assert block.small_kernel_merged is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 10, 64, 64))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test ReparamLargeKernelConv with in_channels == out_channels,
    # kernel_size = 31, stride = 1, groups=in_channels, small_kernel = 5
    block = ReparamLargeKernelConv(
        12, 12, kernel_size=31, stride=1, groups=12, small_kernel=5)
    block.eval()
    x = torch.randn(1, 12, 64, 64)
    x_out_not_deploy = block(x)
    assert block.small_kernel <= block.kernel_size
    assert not hasattr(block, 'lkb_reparam')
    assert hasattr(block, 'lkb_origin')
    assert hasattr(block, 'small_conv')
    assert x_out_not_deploy.shape == torch.Size((1, 12, 64, 64))
    block.merge_kernel()
    assert block.small_kernel_merged is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 12, 64, 64))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test ReparamLargeKernelConv with in_channels == out_channels,
    # kernel_size = 31, stride = 2, groups=in_channels, small_kernel = 5
    block = ReparamLargeKernelConv(
        16, 16, kernel_size=31, stride=2, groups=16, small_kernel=5)
    block.eval()
    x = torch.randn(1, 16, 64, 64)
    x_out_not_deploy = block(x)
    assert block.small_kernel <= block.kernel_size
    assert not hasattr(block, 'lkb_reparam')
    assert hasattr(block, 'lkb_origin')
    assert hasattr(block, 'small_conv')
    assert x_out_not_deploy.shape == torch.Size((1, 16, 32, 32))
    block.merge_kernel()
    assert block.small_kernel_merged is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 16, 32, 32))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test ReparamLargeKernelConv with in_channels == out_channels,
    # kernel_size = 27, stride = 1, groups=in_channels, small_kernel = 5
    block = ReparamLargeKernelConv(
        12, 12, kernel_size=27, stride=1, groups=12, small_kernel=5)
    block.eval()
    x = torch.randn(1, 12, 48, 48)
    x_out_not_deploy = block(x)
    assert block.small_kernel <= block.kernel_size
    assert not hasattr(block, 'lkb_reparam')
    assert hasattr(block, 'lkb_origin')
    assert hasattr(block, 'small_conv')
    assert x_out_not_deploy.shape == torch.Size((1, 12, 48, 48))
    block.merge_kernel()
    assert block.small_kernel_merged is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 12, 48, 48))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test ReparamLargeKernelConv with in_channels == out_channels,
    # kernel_size = 31, stride = 1, groups=in_channels, small_kernel = 7
    block = ReparamLargeKernelConv(
        12, 12, kernel_size=31, stride=1, groups=12, small_kernel=7)
    block.eval()
    x = torch.randn(1, 12, 64, 64)
    x_out_not_deploy = block(x)
    assert block.small_kernel <= block.kernel_size
    assert not hasattr(block, 'lkb_reparam')
    assert hasattr(block, 'lkb_origin')
    assert hasattr(block, 'small_conv')
    assert x_out_not_deploy.shape == torch.Size((1, 12, 64, 64))
    block.merge_kernel()
    assert block.small_kernel_merged is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 12, 64, 64))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test ReparamLargeKernelConv with deploy == True
    block = ReparamLargeKernelConv(
        8,
        8,
        kernel_size=31,
        stride=1,
        groups=8,
        small_kernel=5,
        small_kernel_merged=True)
    assert isinstance(block.lkb_reparam, nn.Conv2d)
    assert not hasattr(block, 'lkb_origin')
    assert not hasattr(block, 'small_conv')
    x = torch.randn(1, 8, 48, 48)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 8, 48, 48))


def test_replknet_backbone():
    with pytest.raises(TypeError):
        # arch must be str or dict
        RepLKNet(arch=[4, 6, 16, 1])

    with pytest.raises(AssertionError):
        # arch must in arch_settings
        RepLKNet(arch='31C')

    with pytest.raises(KeyError):
        # arch must have num_blocks and width_factor
        arch = dict(large_kernel_sizes=[31, 29, 27, 13])
        RepLKNet(arch=arch)

    with pytest.raises(KeyError):
        # arch must have num_blocks and width_factor
        arch = dict(large_kernel_sizes=[31, 29, 27, 13], layers=[2, 2, 18, 2])
        RepLKNet(arch=arch)

    with pytest.raises(KeyError):
        # arch must have num_blocks and width_factor
        arch = dict(
            large_kernel_sizes=[31, 29, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[128, 256, 512, 1024])
        RepLKNet(arch=arch)

    # len(arch['large_kernel_sizes']) == arch['layers'])
    # == len(arch['channels'])
    # == len(strides) == len(dilations)
    with pytest.raises(AssertionError):
        arch = dict(
            large_kernel_sizes=[31, 29, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[128, 256, 1024],
            small_kernel=5,
            dw_ratio=1)
        RepLKNet(arch=arch)

    # len(strides) must equal to 4
    with pytest.raises(AssertionError):
        RepLKNet('31B', strides=(2, 2, 2))

    # len(dilations) must equal to 4
    with pytest.raises(AssertionError):
        RepLKNet('31B', strides=(2, 2, 2, 2), dilations=(1, 1, 1))

    # max(out_indices) < len(arch['num_blocks'])
    with pytest.raises(AssertionError):
        RepLKNet('31B', out_indices=(5, ))

    # Test RepLKNet norm state
    model = RepLKNet('31B')
    model.train()
    assert check_norm_state(model.modules(), True)

    # Test RepLKNet with first stage frozen
    frozen_stages = 1
    model = RepLKNet('31B', frozen_stages=frozen_stages)
    model.train()
    for param in model.stem.parameters():
        assert param.requires_grad is False
    for i in range(0, frozen_stages):
        stage = model.stages[i]
        for mod in stage.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in stage.parameters():
            assert param.requires_grad is False

    # Test RepLKNet with norm_eval
    model = RepLKNet('31B', norm_eval=True)
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test RepLKNet forward with layer 3 forward
    model = RepLKNet('31B', out_indices=(3, ))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert isinstance(feat, tuple)
    assert len(feat) == 1
    assert isinstance(feat[0], torch.Tensor)
    assert feat[0].shape == torch.Size((1, 1024, 7, 7))

    # Test RepLKNet forward
    model_test_settings = [
        dict(model_name='31B', out_sizes=(128, 256, 512, 1024)),
        # dict(model_name='31L', out_sizes=(192, 384, 768, 1536)),
        # dict(model_name='XL', out_sizes=(256, 512, 1024, 2048))
    ]

    choose_models = ['31B']
    # Test RepLKNet model forward
    for model_test_setting in model_test_settings:
        if model_test_setting['model_name'] not in choose_models:
            continue
        model = RepLKNet(
            model_test_setting['model_name'], out_indices=(0, 1, 2, 3))
        model.init_weights()

        # Test Norm
        for m in model.modules():
            if is_norm(m):
                assert isinstance(m, _BatchNorm)

        model.train()
        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        assert feat[0].shape == torch.Size(
            (1, model_test_setting['out_sizes'][0], 56, 56))
        assert feat[1].shape == torch.Size(
            (1, model_test_setting['out_sizes'][1], 28, 28))
        assert feat[2].shape == torch.Size(
            (1, model_test_setting['out_sizes'][2], 14, 14))
        assert feat[3].shape == torch.Size(
            (1, model_test_setting['out_sizes'][3], 7, 7))

        # Test eval of "train" mode and "deploy" mode
        gap = nn.AdaptiveAvgPool2d(output_size=(1))
        fc = nn.Linear(model_test_setting['out_sizes'][3], 10)
        model.eval()
        feat = model(imgs)
        pred = fc(gap(feat[3]).flatten(1))
        model.switch_to_deploy()
        for m in model.modules():
            if isinstance(m, ReparamLargeKernelConv):
                assert m.small_kernel_merged is True
        feat_deploy = model(imgs)
        pred_deploy = fc(gap(feat_deploy[3]).flatten(1))
        for i in range(4):
            torch.allclose(feat[i], feat_deploy[i])
        torch.allclose(pred, pred_deploy)


def test_replknet_load():
    # Test output before and load from deploy checkpoint
    model = RepLKNet('31B', out_indices=(0, 1, 2, 3))
    inputs = torch.randn((1, 3, 224, 224))
    ckpt_path = os.path.join(tempfile.gettempdir(), 'ckpt.pth')
    model.switch_to_deploy()
    model.eval()
    outputs = model(inputs)

    model_deploy = RepLKNet(
        '31B', out_indices=(0, 1, 2, 3), small_kernel_merged=True)
    model_deploy.eval()
    save_checkpoint(model.state_dict(), ckpt_path)
    load_checkpoint(model_deploy, ckpt_path, strict=True)

    outputs_load = model_deploy(inputs)
    for feat, feat_load in zip(outputs, outputs_load):
        assert torch.allclose(feat, feat_load)

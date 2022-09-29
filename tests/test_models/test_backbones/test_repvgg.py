# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile

import pytest
import torch
from mmcv.runner import load_checkpoint, save_checkpoint
from torch import nn
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import RepVGG
from mmcls.models.backbones.repvgg import RepVGGBlock
from mmcls.models.utils import SELayer


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


def is_repvgg_block(modules):
    if isinstance(modules, RepVGGBlock):
        return True
    return False


def test_repvgg_repvggblock():
    # Test RepVGGBlock with in_channels != out_channels, stride = 1
    block = RepVGGBlock(5, 10, stride=1)
    block.eval()
    x = torch.randn(1, 5, 16, 16)
    x_out_not_deploy = block(x)
    assert block.branch_norm is None
    assert not hasattr(block, 'branch_reparam')
    assert hasattr(block, 'branch_1x1')
    assert hasattr(block, 'branch_3x3')
    assert hasattr(block, 'branch_norm')
    assert block.se_cfg is None
    assert x_out_not_deploy.shape == torch.Size((1, 10, 16, 16))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 10, 16, 16))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test RepVGGBlock with in_channels == out_channels, stride = 1
    block = RepVGGBlock(12, 12, stride=1)
    block.eval()
    x = torch.randn(1, 12, 8, 8)
    x_out_not_deploy = block(x)
    assert isinstance(block.branch_norm, nn.BatchNorm2d)
    assert not hasattr(block, 'branch_reparam')
    assert x_out_not_deploy.shape == torch.Size((1, 12, 8, 8))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 12, 8, 8))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test RepVGGBlock with in_channels == out_channels, stride = 2
    block = RepVGGBlock(16, 16, stride=2)
    block.eval()
    x = torch.randn(1, 16, 8, 8)
    x_out_not_deploy = block(x)
    assert block.branch_norm is None
    assert x_out_not_deploy.shape == torch.Size((1, 16, 4, 4))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 16, 4, 4))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test RepVGGBlock with padding == dilation == 2
    block = RepVGGBlock(14, 14, stride=1, padding=2, dilation=2)
    block.eval()
    x = torch.randn(1, 14, 16, 16)
    x_out_not_deploy = block(x)
    assert isinstance(block.branch_norm, nn.BatchNorm2d)
    assert x_out_not_deploy.shape == torch.Size((1, 14, 16, 16))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 14, 16, 16))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test RepVGGBlock with groups = 2
    block = RepVGGBlock(4, 4, stride=1, groups=2)
    block.eval()
    x = torch.randn(1, 4, 5, 6)
    x_out_not_deploy = block(x)
    assert x_out_not_deploy.shape == torch.Size((1, 4, 5, 6))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 4, 5, 6))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-5, rtol=1e-4)

    # Test RepVGGBlock with se
    se_cfg = dict(ratio=4, divisor=1)
    block = RepVGGBlock(18, 18, stride=1, se_cfg=se_cfg)
    block.train()
    x = torch.randn(1, 18, 5, 5)
    x_out_not_deploy = block(x)
    assert isinstance(block.se_layer, SELayer)
    assert x_out_not_deploy.shape == torch.Size((1, 18, 5, 5))

    # Test RepVGGBlock with checkpoint forward
    block = RepVGGBlock(24, 24, stride=1, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 24, 7, 7)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 24, 7, 7))

    # Test RepVGGBlock with deploy == True
    block = RepVGGBlock(8, 8, stride=1, deploy=True)
    assert isinstance(block.branch_reparam, nn.Conv2d)
    assert not hasattr(block, 'branch_3x3')
    assert not hasattr(block, 'branch_1x1')
    assert not hasattr(block, 'branch_norm')
    x = torch.randn(1, 8, 16, 16)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 8, 16, 16))


def test_repvgg_backbone():
    with pytest.raises(TypeError):
        # arch must be str or dict
        RepVGG(arch=[4, 6, 16, 1])

    with pytest.raises(AssertionError):
        # arch must in arch_settings
        RepVGG(arch='A3')

    with pytest.raises(KeyError):
        # arch must have num_blocks and width_factor
        arch = dict(num_blocks=[2, 4, 14, 1])
        RepVGG(arch=arch)

    # len(arch['num_blocks']) == len(arch['width_factor'])
    # == len(strides) == len(dilations)
    with pytest.raises(AssertionError):
        arch = dict(num_blocks=[2, 4, 14, 1], width_factor=[0.75, 0.75, 0.75])
        RepVGG(arch=arch)

    # len(strides) must equal to 4
    with pytest.raises(AssertionError):
        RepVGG('A0', strides=(1, 1, 1))

    # len(dilations) must equal to 4
    with pytest.raises(AssertionError):
        RepVGG('A0', strides=(1, 1, 1, 1), dilations=(1, 1, 2))

    # max(out_indices) < len(arch['num_blocks'])
    with pytest.raises(AssertionError):
        RepVGG('A0', out_indices=(5, ))

    # max(arch['group_idx'].keys()) <= sum(arch['num_blocks'])
    with pytest.raises(AssertionError):
        arch = dict(
            num_blocks=[2, 4, 14, 1],
            width_factor=[0.75, 0.75, 0.75],
            group_idx={22: 2})
        RepVGG(arch=arch)

    # Test RepVGG norm state
    model = RepVGG('A0')
    model.train()
    assert check_norm_state(model.modules(), True)

    # Test RepVGG with first stage frozen
    frozen_stages = 1
    model = RepVGG('A0', frozen_stages=frozen_stages)
    model.train()
    for param in model.stem.parameters():
        assert param.requires_grad is False
    for i in range(0, frozen_stages):
        stage_name = model.stages[i]
        stage = model.__getattr__(stage_name)
        for mod in stage:
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in stage.parameters():
            assert param.requires_grad is False

    # Test RepVGG with norm_eval
    model = RepVGG('A0', norm_eval=True)
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test RepVGG forward with layer 3 forward
    model = RepVGG('A0', out_indices=(3, ))
    model.init_weights()
    model.eval()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert isinstance(feat, tuple)
    assert len(feat) == 1
    assert isinstance(feat[0], torch.Tensor)
    assert feat[0].shape == torch.Size((1, 1280, 1, 1))

    # Test with custom arch
    cfg = dict(
        num_blocks=[3, 5, 7, 3],
        width_factor=[1, 1, 1, 1],
        group_layer_map=None,
        se_cfg=None,
        stem_channels=16)
    model = RepVGG(arch=cfg, out_indices=(3, ))
    model.eval()
    assert model.stem.out_channels == min(16, 64 * 1)

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert isinstance(feat, tuple)
    assert len(feat) == 1
    assert isinstance(feat[0], torch.Tensor)
    assert feat[0].shape == torch.Size((1, 512, 1, 1))

    # Test RepVGG forward
    model_test_settings = [
        dict(model_name='A0', out_sizes=(48, 96, 192, 1280)),
        dict(model_name='A1', out_sizes=(64, 128, 256, 1280)),
        dict(model_name='A2', out_sizes=(96, 192, 384, 1408)),
        dict(model_name='B0', out_sizes=(64, 128, 256, 1280)),
        dict(model_name='B1', out_sizes=(128, 256, 512, 2048)),
        dict(model_name='B1g2', out_sizes=(128, 256, 512, 2048)),
        dict(model_name='B1g4', out_sizes=(128, 256, 512, 2048)),
        dict(model_name='B2', out_sizes=(160, 320, 640, 2560)),
        dict(model_name='B2g2', out_sizes=(160, 320, 640, 2560)),
        dict(model_name='B2g4', out_sizes=(160, 320, 640, 2560)),
        dict(model_name='B3', out_sizes=(192, 384, 768, 2560)),
        dict(model_name='B3g2', out_sizes=(192, 384, 768, 2560)),
        dict(model_name='B3g4', out_sizes=(192, 384, 768, 2560)),
        dict(model_name='D2se', out_sizes=(160, 320, 640, 2560))
    ]

    choose_models = ['A0', 'B1', 'B1g2']
    # Test RepVGG model forward
    for model_test_setting in model_test_settings:
        if model_test_setting['model_name'] not in choose_models:
            continue
        model = RepVGG(
            model_test_setting['model_name'], out_indices=(0, 1, 2, 3))
        model.init_weights()
        model.eval()

        # Test Norm
        for m in model.modules():
            if is_norm(m):
                assert isinstance(m, _BatchNorm)

        imgs = torch.randn(1, 3, 32, 32)
        feat = model(imgs)
        assert feat[0].shape == torch.Size(
            (1, model_test_setting['out_sizes'][0], 8, 8))
        assert feat[1].shape == torch.Size(
            (1, model_test_setting['out_sizes'][1], 4, 4))
        assert feat[2].shape == torch.Size(
            (1, model_test_setting['out_sizes'][2], 2, 2))
        assert feat[3].shape == torch.Size(
            (1, model_test_setting['out_sizes'][3], 1, 1))

        # Test eval of "train" mode and "deploy" mode
        gap = nn.AdaptiveAvgPool2d(output_size=(1))
        fc = nn.Linear(model_test_setting['out_sizes'][3], 10)
        model.eval()
        feat = model(imgs)
        pred = fc(gap(feat[3]).flatten(1))
        model.switch_to_deploy()
        for m in model.modules():
            if isinstance(m, RepVGGBlock):
                assert m.deploy is True
        feat_deploy = model(imgs)
        pred_deploy = fc(gap(feat_deploy[3]).flatten(1))
        for i in range(4):
            torch.allclose(feat[i], feat_deploy[i])
        torch.allclose(pred, pred_deploy)

    # Test RepVGG forward with add_ppf
    model = RepVGG('A0', out_indices=(3, ), add_ppf=True)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 64, 64)
    feat = model(imgs)
    assert isinstance(feat, tuple)
    assert len(feat) == 1
    assert isinstance(feat[0], torch.Tensor)
    assert feat[0].shape == torch.Size((1, 1280, 2, 2))

    # Test RepVGG forward with 'stem_channels' not in arch
    arch = dict(
        num_blocks=[2, 4, 14, 1],
        width_factor=[0.75, 0.75, 0.75, 2.5],
        group_layer_map=None,
        se_cfg=None)
    model = RepVGG(arch, add_ppf=True)
    model.stem.in_channels = min(64, 64 * 0.75)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 64, 64)
    feat = model(imgs)
    assert isinstance(feat, tuple)
    assert len(feat) == 1
    assert isinstance(feat[0], torch.Tensor)
    assert feat[0].shape == torch.Size((1, 1280, 2, 2))


def test_repvgg_load():
    # Test output before and load from deploy checkpoint
    model = RepVGG('A1', out_indices=(0, 1, 2, 3))
    inputs = torch.randn((1, 3, 32, 32))
    ckpt_path = os.path.join(tempfile.gettempdir(), 'ckpt.pth')
    model.switch_to_deploy()
    model.eval()
    outputs = model(inputs)

    model_deploy = RepVGG('A1', out_indices=(0, 1, 2, 3), deploy=True)
    save_checkpoint(model, ckpt_path)
    load_checkpoint(model_deploy, ckpt_path, strict=True)

    outputs_load = model_deploy(inputs)
    for feat, feat_load in zip(outputs, outputs_load):
        assert torch.allclose(feat, feat_load)

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
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-6, rtol=1e-5)

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
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-6, rtol=1e-5)

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
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-6, rtol=1e-5)

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
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-6, rtol=1e-5)

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
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-6, rtol=1e-5)

    # Test RepVGGBlock with se
    se_cfg = dict(ratio=4, divisor=1)
    block = RepVGGBlock(18, 18, stride=1, se_cfg=se_cfg)
    block.eval()
    x = torch.randn(1, 18, 5, 5)
    x_out_not_deploy = block(x)
    assert isinstance(block.se_layer, SELayer)
    assert x_out_not_deploy.shape == torch.Size((1, 18, 5, 5))
    block.switch_to_deploy()
    assert block.deploy is True
    x_out_deploy = block(x)
    assert x_out_deploy.shape == torch.Size((1, 18, 5, 5))
    assert torch.allclose(x_out_not_deploy, x_out_deploy, atol=1e-6, rtol=1e-5)

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
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    # Test RepVGG with first stage frozen
    frozen_stages = 1
    model = RepVGG('A0', frozen_stages=frozen_stages)
    model.init_weights()
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
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test RepVGG forward with arch='A0'
    model = RepVGG('A0', out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size((1, 48, 56, 56))
    assert feat[1].shape == torch.Size((1, 96, 28, 28))
    assert feat[2].shape == torch.Size((1, 192, 14, 14))
    assert feat[3].shape == torch.Size((1, 1280, 7, 7))

    # Test RepVGG forward with layer 3 forward
    model = RepVGG('A0', out_indices=(3, ))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert isinstance(feat, torch.Tensor)
    assert feat.shape == torch.Size((1, 1280, 7, 7))

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

    choose_models = ['A1', 'B0', 'B1g2', 'B2', 'B3g2', 'D2se']
    # Test RepVGG model
    for model_test_setting in model_test_settings:
        if model_test_setting['model_name'] not in choose_models:
            continue
        model = RepVGG(
            model_test_setting['model_name'], out_indices=(0, 1, 2, 3))
        model.init_weights()
        model.train()

        for m in model.modules():
            if is_norm(m):
                assert isinstance(m, _BatchNorm)

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


def test_repvgg_deploy():
    model = RepVGG('A0', out_indices=(0, 1, 2, 3))
    model.eval()

    # Test ouput before and after deploy
    inputs = torch.randn((1, 3, 224, 224))
    outputs_before_deploy = model(inputs)
    model.switch_to_deploy()
    outputs_after_deploy = model(inputs)

    for features_before_deploy, features_after_deploy in zip(
            outputs_before_deploy, outputs_after_deploy):
        assert torch.allclose(
            features_before_deploy,
            features_after_deploy,
            atol=1e-4,
            rtol=1e-4)

    # Test deploy model load from checkpoint
    ckpt_path = os.path.join(tempfile.gettempdir(), 'ckpt.pth')
    save_checkpoint(model, ckpt_path)
    model_deploy = RepVGG('A0', out_indices=(0, 1, 2, 3), deploy=True)
    load_checkpoint(model_deploy, ckpt_path, strict=True)
    outputs_load = model(inputs)
    for features_after_deploy, features_load in zip(outputs_after_deploy,
                                                    outputs_load):
        assert torch.allclose(features_after_deploy, features_load)

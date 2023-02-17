# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile

import pytest
import torch
from mmengine.runner import load_checkpoint, save_checkpoint
from torch import nn
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import MobileOne
from mmpretrain.models.backbones.mobileone import MobileOneBlock
from mmpretrain.models.utils import SELayer


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


def is_mobileone_block(modules):
    if isinstance(modules, MobileOneBlock):
        return True
    return False


def test_mobileoneblock():
    # Test MobileOneBlock with kernel_size 3
    block = MobileOneBlock(5, 10, 3, 1, stride=1, groups=5)
    block.eval()
    x = torch.randn(1, 5, 16, 16)
    y = block(x)
    assert block.branch_norm is None
    assert not hasattr(block, 'branch_reparam')
    assert hasattr(block, 'branch_scale')
    assert hasattr(block, 'branch_conv_list')
    assert hasattr(block, 'branch_norm')
    assert block.branch_conv_list[0].conv.kernel_size == (3, 3)
    assert block.branch_conv_list[0].conv.groups == 5
    assert block.se_cfg is None
    assert y.shape == torch.Size((1, 10, 16, 16))
    block.switch_to_deploy()
    assert hasattr(block, 'branch_reparam')
    assert block.branch_reparam.kernel_size == (3, 3)
    assert block.branch_reparam.groups == 5
    assert block.deploy is True
    y_deploy = block(x)
    assert y_deploy.shape == torch.Size((1, 10, 16, 16))
    assert torch.allclose(y, y_deploy, atol=1e-5, rtol=1e-4)

    # Test MobileOneBlock with num_con = 4
    block = MobileOneBlock(5, 10, 3, 4, stride=1, groups=5)
    block.eval()
    x = torch.randn(1, 5, 16, 16)
    y = block(x)
    assert block.branch_norm is None
    assert not hasattr(block, 'branch_reparam')
    assert hasattr(block, 'branch_scale')
    assert hasattr(block, 'branch_conv_list')
    assert hasattr(block, 'branch_norm')
    assert block.branch_conv_list[0].conv.kernel_size == (3, 3)
    assert block.branch_conv_list[0].conv.groups == 5
    assert len(block.branch_conv_list) == 4
    assert block.se_cfg is None
    assert y.shape == torch.Size((1, 10, 16, 16))
    block.switch_to_deploy()
    assert hasattr(block, 'branch_reparam')
    assert block.branch_reparam.kernel_size == (3, 3)
    assert block.branch_reparam.groups == 5
    assert block.deploy is True
    y_deploy = block(x)
    assert y_deploy.shape == torch.Size((1, 10, 16, 16))
    assert torch.allclose(y, y_deploy, atol=1e-5, rtol=1e-4)

    # Test MobileOneBlock with kernel_size 1
    block = MobileOneBlock(5, 10, 1, 1, stride=1, padding=0)
    block.eval()
    x = torch.randn(1, 5, 16, 16)
    y = block(x)
    assert block.branch_norm is None
    assert not hasattr(block, 'branch_reparam')
    assert hasattr(block, 'branch_scale')
    assert hasattr(block, 'branch_conv_list')
    assert hasattr(block, 'branch_norm')
    assert block.branch_conv_list[0].conv.kernel_size == (1, 1)
    assert block.branch_conv_list[0].conv.groups == 1
    assert len(block.branch_conv_list) == 1
    assert block.se_cfg is None
    assert y.shape == torch.Size((1, 10, 16, 16))
    block.switch_to_deploy()
    assert hasattr(block, 'branch_reparam')
    assert block.branch_reparam.kernel_size == (1, 1)
    assert block.branch_reparam.groups == 1
    assert block.deploy is True
    y_deploy = block(x)
    assert y_deploy.shape == torch.Size((1, 10, 16, 16))
    assert torch.allclose(y, y_deploy, atol=1e-5, rtol=1e-4)

    # Test MobileOneBlock with stride = 2
    block = MobileOneBlock(10, 10, 3, 4, stride=2, groups=10)
    x = torch.randn(1, 10, 16, 16)
    block.eval()
    y = block(x)
    assert block.branch_norm is None
    assert not hasattr(block, 'branch_reparam')
    assert hasattr(block, 'branch_scale')
    assert hasattr(block, 'branch_conv_list')
    assert hasattr(block, 'branch_norm')
    assert block.branch_conv_list[0].conv.kernel_size == (3, 3)
    assert block.branch_conv_list[0].conv.groups == 10
    assert len(block.branch_conv_list) == 4
    assert block.se_cfg is None
    assert y.shape == torch.Size((1, 10, 8, 8))
    block.switch_to_deploy()
    assert hasattr(block, 'branch_reparam')
    assert block.branch_reparam.kernel_size == (3, 3)
    assert block.branch_reparam.groups == 10
    assert block.deploy is True
    y_deploy = block(x)
    assert y_deploy.shape == torch.Size((1, 10, 8, 8))
    assert torch.allclose(y, y_deploy, atol=1e-5, rtol=1e-4)

    # # Test MobileOneBlock with padding == dilation == 2
    block = MobileOneBlock(
        10, 10, 3, 4, stride=1, groups=10, padding=2, dilation=2)
    x = torch.randn(1, 10, 16, 16)
    block.eval()
    y = block(x)
    assert not hasattr(block, 'branch_reparam')
    assert hasattr(block, 'branch_scale')
    assert hasattr(block, 'branch_conv_list')
    assert hasattr(block, 'branch_norm')
    assert block.branch_conv_list[0].conv.kernel_size == (3, 3)
    assert block.branch_conv_list[0].conv.groups == 10
    assert len(block.branch_conv_list) == 4
    assert block.se_cfg is None
    assert y.shape == torch.Size((1, 10, 16, 16))
    block.switch_to_deploy()
    assert hasattr(block, 'branch_reparam')
    assert block.branch_reparam.kernel_size == (3, 3)
    assert block.branch_reparam.groups == 10
    assert block.deploy is True
    y_deploy = block(x)
    assert y_deploy.shape == torch.Size((1, 10, 16, 16))
    assert torch.allclose(y, y_deploy, atol=1e-5, rtol=1e-4)

    # Test MobileOneBlock with se
    se_cfg = dict(ratio=4, divisor=1)
    block = MobileOneBlock(32, 32, 3, 4, stride=1, se_cfg=se_cfg, groups=32)
    x = torch.randn(1, 32, 16, 16)
    block.eval()
    y = block(x)
    assert not hasattr(block, 'branch_reparam')
    assert hasattr(block, 'branch_scale')
    assert hasattr(block, 'branch_conv_list')
    assert hasattr(block, 'branch_norm')
    assert block.branch_conv_list[0].conv.kernel_size == (3, 3)
    assert block.branch_conv_list[0].conv.groups == 32
    assert len(block.branch_conv_list) == 4
    assert isinstance(block.se, SELayer)
    assert y.shape == torch.Size((1, 32, 16, 16))
    block.switch_to_deploy()
    assert hasattr(block, 'branch_reparam')
    assert block.branch_reparam.kernel_size == (3, 3)
    assert block.branch_reparam.groups == 32
    assert block.deploy is True
    y_deploy = block(x)
    assert y_deploy.shape == torch.Size((1, 32, 16, 16))
    assert torch.allclose(y, y_deploy, atol=1e-5, rtol=1e-4)

    # Test MobileOneBlock with deploy == True
    se_cfg = dict(ratio=4, divisor=1)
    block = MobileOneBlock(
        32, 32, 3, 4, stride=1, se_cfg=se_cfg, groups=32, deploy=True)
    x = torch.randn(1, 32, 16, 16)
    block.eval()
    assert hasattr(block, 'branch_reparam')
    assert block.branch_reparam.kernel_size == (3, 3)
    assert block.branch_reparam.groups == 32
    assert isinstance(block.se, SELayer)
    assert block.deploy is True
    y = block(x)
    assert y.shape == torch.Size((1, 32, 16, 16))


def test_mobileone_backbone():
    with pytest.raises(TypeError):
        # arch must be str or dict
        MobileOne(arch=[4, 6, 16, 1])

    with pytest.raises(AssertionError):
        # arch must in arch_settings
        MobileOne(arch='S3')

    with pytest.raises(KeyError):
        arch = dict(num_blocks=[2, 4, 14, 1])
        MobileOne(arch=arch)

    # Test  len(arch['num_blocks']) == len(arch['width_factor'])
    with pytest.raises(AssertionError):
        arch = dict(
            num_blocks=[2, 4, 14, 1],
            width_factor=[0.75, 0.75, 0.75],
            num_conv_branches=[1, 1, 1, 1],
            num_se_blocks=[0, 0, 5, 1])
        MobileOne(arch=arch)

    # Test max(out_indices) < len(arch['num_blocks'])
    with pytest.raises(AssertionError):
        MobileOne('s0', out_indices=dict())

    # Test out_indices not type of int or Sequence
    with pytest.raises(AssertionError):
        MobileOne('s0', out_indices=(5, ))

    # Test MobileOne norm state
    model = MobileOne('s0')
    model.train()
    assert check_norm_state(model.modules(), True)

    # Test MobileOne with first stage frozen
    frozen_stages = 1
    model = MobileOne('s0', frozen_stages=frozen_stages)
    model.train()
    for param in model.stage0.parameters():
        assert param.requires_grad is False
    for i in range(0, frozen_stages):
        stage_name = model.stages[i]
        stage = model.__getattr__(stage_name)
        for mod in stage:
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in stage.parameters():
            assert param.requires_grad is False

    # Test MobileOne with norm_eval
    model = MobileOne('s0', norm_eval=True)
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test MobileOne forward with layer 3 forward
    model = MobileOne('s0', out_indices=(3, ))
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

    # Test MobileOne forward
    arch_settings = {
        's0': dict(out_channels=[48, 128, 256, 1024], ),
        's1': dict(out_channels=[96, 192, 512, 1280]),
        's2': dict(out_channels=[96, 256, 640, 2048]),
        's3': dict(out_channels=[128, 320, 768, 2048], ),
        's4': dict(out_channels=[192, 448, 896, 2048], )
    }

    choose_models = ['s0', 's1', 's4']
    # Test RepVGG model forward
    for model_name, model_arch in arch_settings.items():
        if model_name not in choose_models:
            continue
        model = MobileOne(model_name, out_indices=(0, 1, 2, 3))
        model.init_weights()

        # Test Norm
        for m in model.modules():
            if is_norm(m):
                assert isinstance(m, _BatchNorm)

        model.train()
        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        assert feat[0].shape == torch.Size(
            (1, model_arch['out_channels'][0], 56, 56))
        assert feat[1].shape == torch.Size(
            (1, model_arch['out_channels'][1], 28, 28))
        assert feat[2].shape == torch.Size(
            (1, model_arch['out_channels'][2], 14, 14))
        assert feat[3].shape == torch.Size(
            (1, model_arch['out_channels'][3], 7, 7))

        # Test eval of "train" mode and "deploy" mode
        gap = nn.AdaptiveAvgPool2d(output_size=(1))
        fc = nn.Linear(model_arch['out_channels'][3], 10)
        model.eval()
        feat = model(imgs)
        pred = fc(gap(feat[3]).flatten(1))
        model.switch_to_deploy()
        for m in model.modules():
            if isinstance(m, MobileOneBlock):
                assert m.deploy is True
        feat_deploy = model(imgs)
        pred_deploy = fc(gap(feat_deploy[3]).flatten(1))
        for i in range(4):
            torch.allclose(feat[i], feat_deploy[i])
        torch.allclose(pred, pred_deploy)


def test_load_deploy_mobileone():
    # Test output before and load from deploy checkpoint
    model = MobileOne('s0', out_indices=(0, 1, 2, 3))
    inputs = torch.randn((1, 3, 224, 224))
    tmpdir = tempfile.gettempdir()
    ckpt_path = os.path.join(tmpdir, 'ckpt.pth')
    model.switch_to_deploy()
    model.eval()
    outputs = model(inputs)

    model_deploy = MobileOne('s0', out_indices=(0, 1, 2, 3), deploy=True)
    save_checkpoint(model.state_dict(), ckpt_path)
    load_checkpoint(model_deploy, ckpt_path)

    outputs_load = model_deploy(inputs)
    for feat, feat_load in zip(outputs, outputs_load):
        assert torch.allclose(feat, feat_load)
    os.remove(ckpt_path)

# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile

import pytest
import torch
from mmengine.runner import load_checkpoint, save_checkpoint
from torch import nn
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import levit
from mmpretrain.models.backbones.levit import (Attention, AttentionSubsample,
                                               LeViT)


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


def is_levit_block(modules):
    if isinstance(modules, (AttentionSubsample, Attention)):
        return True
    return False


def test_levit_attention():
    block = Attention(128, 16, 4, 2, act_cfg=dict(type='HSwish'))
    block.eval()
    x = torch.randn(1, 196, 128)
    y = block(x)
    assert y.shape == x.shape
    assert hasattr(block, 'ab')
    assert block.key_dim == 16
    assert block.attn_ratio == 2
    assert block.num_heads == 4
    assert block.qkv.linear.in_features == 128


def test_levit():
    with pytest.raises(TypeError):
        # arch must be str or dict
        LeViT(arch=[4, 6, 16, 1])

    with pytest.raises(AssertionError):
        # arch must in arch_settings
        LeViT(arch='512')

    with pytest.raises(AssertionError):
        arch = dict(num_blocks=[2, 4, 14, 1])
        LeViT(arch=arch)

    # Test out_indices not type of int or Sequence
    with pytest.raises(TypeError):
        LeViT('128s', out_indices=dict())

    # Test max(out_indices) < len(arch['num_blocks'])
    with pytest.raises(AssertionError):
        LeViT('128s', out_indices=(3, ))

    model = LeViT('128s', out_indices=(-1, ))
    assert model.out_indices == [2]

    model = LeViT(arch='256', drop_path_rate=0.1)
    model.eval()
    assert model.key_dims == [32, 32, 32]
    assert model.embed_dims == [256, 384, 512]
    assert model.num_heads == [4, 6, 8]
    assert model.depths == [4, 4, 4]
    assert model.drop_path_rate == 0.1
    assert isinstance(model.stages[0][0].block.qkv, levit.LinearBatchNorm)
    assert isinstance(model.patch_embed.patch_embed[0],
                      levit.ConvolutionBatchNorm)

    model = LeViT(
        arch='128s',
        hybrid_backbone=lambda embed_dims: nn.Conv2d(
            embed_dims, embed_dims, kernel_size=2))
    model.eval()
    assert isinstance(model.patch_embed, nn.Conv2d)

    # Test eval of "train" mode and "deploy" mode
    model = LeViT(arch='128s', deploy=True)
    model.eval()
    assert not isinstance(model.stages[0][0].block.qkv, levit.LinearBatchNorm)
    assert not isinstance(model.patch_embed.patch_embed[0],
                          levit.ConvolutionBatchNorm)
    assert isinstance(model.stages[0][0].block.qkv, nn.Linear)
    assert isinstance(model.patch_embed.patch_embed[0], nn.Conv2d)

    # Test LeViT forward with layer 2 forward
    model = LeViT('128s', out_indices=(2, ))
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
    assert feat[0].shape == torch.Size((1, 384, 4, 4))

    # Test LeViT forward
    arch_settings = {
        '128s': dict(out_channels=[128, 256, 384]),
        '128': dict(out_channels=[128, 256, 384]),
        '192': dict(out_channels=[192, 288, 384]),
        '256': dict(out_channels=[256, 384, 512]),
        '384': dict(out_channels=[384, 512, 768])
    }

    choose_models = ['128s', '192', '256', '384']
    # Test LeViT model forward
    for model_name, model_arch in arch_settings.items():
        if model_name not in choose_models:
            continue
        model = LeViT(model_name, out_indices=(0, 1, 2))
        model.init_weights()

        # Test Norm
        for m in model.modules():
            if is_norm(m):
                assert isinstance(m, _BatchNorm)

        model.train()
        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        assert feat[0].shape == torch.Size(
            (1, model_arch['out_channels'][0], 14, 14))
        assert feat[1].shape == torch.Size(
            (1, model_arch['out_channels'][1], 7, 7))
        assert feat[2].shape == torch.Size(
            (1, model_arch['out_channels'][2], 4, 4))


def test_load_deploy_LeViT():
    # Test output before and load from deploy checkpoint
    model = LeViT('128s', out_indices=(0, 1, 2))
    inputs = torch.randn((1, 3, 224, 224))
    tmpdir = tempfile.gettempdir()
    ckpt_path = os.path.join(tmpdir, 'ckpt.pth')
    model.switch_to_deploy()
    model.eval()
    outputs = model(inputs)

    model_deploy = LeViT('128s', out_indices=(0, 1, 2), deploy=True)
    save_checkpoint(model.state_dict(), ckpt_path)
    load_checkpoint(model_deploy, ckpt_path)

    outputs_load = model_deploy(inputs)
    for feat, feat_load in zip(outputs, outputs_load):
        assert torch.allclose(feat, feat_load)
    os.remove(ckpt_path)

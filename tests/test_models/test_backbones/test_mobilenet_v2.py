# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import MobileNetV2
from mmpretrain.models.backbones.mobilenet_v2 import InvertedResidual


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (InvertedResidual, )):
        return True
    return False


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


def test_mobilenetv2_invertedresidual():

    with pytest.raises(AssertionError):
        # stride must be in [1, 2]
        InvertedResidual(16, 24, stride=3, expand_ratio=6)

    # Test InvertedResidual with checkpoint forward, stride=1
    block = InvertedResidual(16, 24, stride=1, expand_ratio=6)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 24, 56, 56))

    # Test InvertedResidual with expand_ratio=1
    block = InvertedResidual(16, 16, stride=1, expand_ratio=1)
    assert len(block.conv) == 2

    # Test InvertedResidual with use_res_connect
    block = InvertedResidual(16, 16, stride=1, expand_ratio=6)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert block.use_res_connect is True
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual with checkpoint forward, stride=2
    block = InvertedResidual(16, 24, stride=2, expand_ratio=6)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 24, 28, 28))

    # Test InvertedResidual with checkpoint forward
    block = InvertedResidual(16, 24, stride=1, expand_ratio=6, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 24, 56, 56))

    # Test InvertedResidual with act_cfg=dict(type='ReLU')
    block = InvertedResidual(
        16, 24, stride=1, expand_ratio=6, act_cfg=dict(type='ReLU'))
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 24, 56, 56))


def test_mobilenetv2_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = MobileNetV2()
        model.init_weights(pretrained=0)

    with pytest.raises(ValueError):
        # frozen_stages must in range(-1, 8)
        MobileNetV2(frozen_stages=8)

    with pytest.raises(ValueError):
        # out_indices in range(0, 8)
        MobileNetV2(out_indices=[8])

    # Test MobileNetV2 with first stage frozen
    frozen_stages = 1
    model = MobileNetV2(frozen_stages=frozen_stages)
    model.init_weights()
    model.train()

    for mod in model.conv1.modules():
        for param in mod.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test MobileNetV2 with norm_eval=True
    model = MobileNetV2(norm_eval=True)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), False)

    # Test MobileNetV2 forward with widen_factor=1.0
    model = MobileNetV2(widen_factor=1.0, out_indices=range(0, 8))
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 8
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 24, 56, 56))
    assert feat[2].shape == torch.Size((1, 32, 28, 28))
    assert feat[3].shape == torch.Size((1, 64, 14, 14))
    assert feat[4].shape == torch.Size((1, 96, 14, 14))
    assert feat[5].shape == torch.Size((1, 160, 7, 7))
    assert feat[6].shape == torch.Size((1, 320, 7, 7))
    assert feat[7].shape == torch.Size((1, 1280, 7, 7))

    # Test MobileNetV2 forward with widen_factor=0.5
    model = MobileNetV2(widen_factor=0.5, out_indices=range(0, 7))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size((1, 8, 112, 112))
    assert feat[1].shape == torch.Size((1, 16, 56, 56))
    assert feat[2].shape == torch.Size((1, 16, 28, 28))
    assert feat[3].shape == torch.Size((1, 32, 14, 14))
    assert feat[4].shape == torch.Size((1, 48, 14, 14))
    assert feat[5].shape == torch.Size((1, 80, 7, 7))
    assert feat[6].shape == torch.Size((1, 160, 7, 7))

    # Test MobileNetV2 forward with widen_factor=2.0
    model = MobileNetV2(widen_factor=2.0)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size((1, 2560, 7, 7))

    # Test MobileNetV2 forward with out_indices=None
    model = MobileNetV2(widen_factor=1.0)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size((1, 1280, 7, 7))

    # Test MobileNetV2 forward with dict(type='ReLU')
    model = MobileNetV2(
        widen_factor=1.0, act_cfg=dict(type='ReLU'), out_indices=range(0, 7))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 24, 56, 56))
    assert feat[2].shape == torch.Size((1, 32, 28, 28))
    assert feat[3].shape == torch.Size((1, 64, 14, 14))
    assert feat[4].shape == torch.Size((1, 96, 14, 14))
    assert feat[5].shape == torch.Size((1, 160, 7, 7))
    assert feat[6].shape == torch.Size((1, 320, 7, 7))

    # Test MobileNetV2 with BatchNorm forward
    model = MobileNetV2(widen_factor=1.0, out_indices=range(0, 7))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 24, 56, 56))
    assert feat[2].shape == torch.Size((1, 32, 28, 28))
    assert feat[3].shape == torch.Size((1, 64, 14, 14))
    assert feat[4].shape == torch.Size((1, 96, 14, 14))
    assert feat[5].shape == torch.Size((1, 160, 7, 7))
    assert feat[6].shape == torch.Size((1, 320, 7, 7))

    # Test MobileNetV2 with GroupNorm forward
    model = MobileNetV2(
        widen_factor=1.0,
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True),
        out_indices=range(0, 7))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 24, 56, 56))
    assert feat[2].shape == torch.Size((1, 32, 28, 28))
    assert feat[3].shape == torch.Size((1, 64, 14, 14))
    assert feat[4].shape == torch.Size((1, 96, 14, 14))
    assert feat[5].shape == torch.Size((1, 160, 7, 7))
    assert feat[6].shape == torch.Size((1, 320, 7, 7))

    # Test MobileNetV2 with layers 1, 3, 5 out forward
    model = MobileNetV2(widen_factor=1.0, out_indices=(0, 2, 4))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 32, 28, 28))
    assert feat[2].shape == torch.Size((1, 96, 14, 14))

    # Test MobileNetV2 with checkpoint forward
    model = MobileNetV2(
        widen_factor=1.0, with_cp=True, out_indices=range(0, 7))
    for m in model.modules():
        if is_block(m):
            assert m.with_cp
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 24, 56, 56))
    assert feat[2].shape == torch.Size((1, 32, 28, 28))
    assert feat[3].shape == torch.Size((1, 64, 14, 14))
    assert feat[4].shape == torch.Size((1, 96, 14, 14))
    assert feat[5].shape == torch.Size((1, 160, 7, 7))
    assert feat[6].shape == torch.Size((1, 320, 7, 7))

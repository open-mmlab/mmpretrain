# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import EfficientNet


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


def test_efficientnet_backbone():
    archs = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b7', 'b8', 'es', 'em', 'el']
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = EfficientNet()
        model.init_weights(pretrained=0)

    with pytest.raises(AssertionError):
        # arch must in arc_settings
        EfficientNet(arch='others')

    for arch in archs:
        with pytest.raises(ValueError):
            # frozen_stages must less than 7
            EfficientNet(arch=arch, frozen_stages=12)

    # Test EfficientNet
    model = EfficientNet()
    model.init_weights()
    model.train()

    # Test EfficientNet with first stage frozen
    frozen_stages = 7
    model = EfficientNet(arch='b0', frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    for i in range(frozen_stages):
        layer = model.layers[i]
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test EfficientNet with norm eval
    model = EfficientNet(norm_eval=True)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test EfficientNet forward with 'b0' arch
    out_channels = [32, 16, 24, 40, 112, 320, 1280]
    model = EfficientNet(arch='b0', out_indices=(0, 1, 2, 3, 4, 5, 6))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size([1, out_channels[0], 112, 112])
    assert feat[1].shape == torch.Size([1, out_channels[1], 112, 112])
    assert feat[2].shape == torch.Size([1, out_channels[2], 56, 56])
    assert feat[3].shape == torch.Size([1, out_channels[3], 28, 28])
    assert feat[4].shape == torch.Size([1, out_channels[4], 14, 14])
    assert feat[5].shape == torch.Size([1, out_channels[5], 7, 7])
    assert feat[6].shape == torch.Size([1, out_channels[6], 7, 7])

    # Test EfficientNet forward with 'b0' arch and GroupNorm
    out_channels = [32, 16, 24, 40, 112, 320, 1280]
    model = EfficientNet(
        arch='b0',
        out_indices=(0, 1, 2, 3, 4, 5, 6),
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size([1, out_channels[0], 112, 112])
    assert feat[1].shape == torch.Size([1, out_channels[1], 112, 112])
    assert feat[2].shape == torch.Size([1, out_channels[2], 56, 56])
    assert feat[3].shape == torch.Size([1, out_channels[3], 28, 28])
    assert feat[4].shape == torch.Size([1, out_channels[4], 14, 14])
    assert feat[5].shape == torch.Size([1, out_channels[5], 7, 7])
    assert feat[6].shape == torch.Size([1, out_channels[6], 7, 7])

    # Test EfficientNet forward with 'es' arch
    out_channels = [32, 24, 32, 48, 144, 192, 1280]
    model = EfficientNet(arch='es', out_indices=(0, 1, 2, 3, 4, 5, 6))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size([1, out_channels[0], 112, 112])
    assert feat[1].shape == torch.Size([1, out_channels[1], 112, 112])
    assert feat[2].shape == torch.Size([1, out_channels[2], 56, 56])
    assert feat[3].shape == torch.Size([1, out_channels[3], 28, 28])
    assert feat[4].shape == torch.Size([1, out_channels[4], 14, 14])
    assert feat[5].shape == torch.Size([1, out_channels[5], 7, 7])
    assert feat[6].shape == torch.Size([1, out_channels[6], 7, 7])

    # Test EfficientNet forward with 'es' arch and GroupNorm
    out_channels = [32, 24, 32, 48, 144, 192, 1280]
    model = EfficientNet(
        arch='es',
        out_indices=(0, 1, 2, 3, 4, 5, 6),
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size([1, out_channels[0], 112, 112])
    assert feat[1].shape == torch.Size([1, out_channels[1], 112, 112])
    assert feat[2].shape == torch.Size([1, out_channels[2], 56, 56])
    assert feat[3].shape == torch.Size([1, out_channels[3], 28, 28])
    assert feat[4].shape == torch.Size([1, out_channels[4], 14, 14])
    assert feat[5].shape == torch.Size([1, out_channels[5], 7, 7])
    assert feat[6].shape == torch.Size([1, out_channels[6], 7, 7])

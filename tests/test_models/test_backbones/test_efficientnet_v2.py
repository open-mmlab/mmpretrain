# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import EfficientNetV2


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


def test_efficientnet_v2_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = EfficientNetV2()
        model.init_weights(pretrained=0)

    with pytest.raises(AssertionError):
        # arch must in arc_settings
        EfficientNetV2(arch='others')

    with pytest.raises(ValueError):
        # frozen_stages must less than 8
        EfficientNetV2(arch='b1', frozen_stages=12)

    # Test EfficientNetV2
    model = EfficientNetV2()
    model.init_weights()
    model.train()
    x = torch.rand((1, 3, 224, 224))
    model(x)

    # Test EfficientNetV2 with first stage frozen
    frozen_stages = 7
    model = EfficientNetV2(arch='b0', frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    for i in range(frozen_stages):
        layer = model.layers[i]
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test EfficientNetV2 with norm eval
    model = EfficientNetV2(norm_eval=True)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test EfficientNetV2 forward with 'b0' arch
    out_channels = [32, 16, 32, 48, 96, 112, 192, 1280]
    model = EfficientNetV2(arch='b0', out_indices=(0, 1, 2, 3, 4, 5, 6, 7))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 8
    assert feat[0].shape == torch.Size([1, out_channels[0], 112, 112])
    assert feat[1].shape == torch.Size([1, out_channels[1], 112, 112])
    assert feat[2].shape == torch.Size([1, out_channels[2], 56, 56])
    assert feat[3].shape == torch.Size([1, out_channels[3], 28, 28])
    assert feat[4].shape == torch.Size([1, out_channels[4], 14, 14])
    assert feat[5].shape == torch.Size([1, out_channels[5], 14, 14])
    assert feat[6].shape == torch.Size([1, out_channels[6], 7, 7])
    assert feat[6].shape == torch.Size([1, out_channels[6], 7, 7])

    # Test EfficientNetV2 forward with 'b0' arch and GroupNorm
    out_channels = [32, 16, 32, 48, 96, 112, 192, 1280]
    model = EfficientNetV2(
        arch='b0',
        out_indices=(0, 1, 2, 3, 4, 5, 6, 7),
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 64, 64)
    feat = model(imgs)
    assert len(feat) == 8
    assert feat[0].shape == torch.Size([1, out_channels[0], 32, 32])
    assert feat[1].shape == torch.Size([1, out_channels[1], 32, 32])
    assert feat[2].shape == torch.Size([1, out_channels[2], 16, 16])
    assert feat[3].shape == torch.Size([1, out_channels[3], 8, 8])
    assert feat[4].shape == torch.Size([1, out_channels[4], 4, 4])
    assert feat[5].shape == torch.Size([1, out_channels[5], 4, 4])
    assert feat[6].shape == torch.Size([1, out_channels[6], 2, 2])
    assert feat[7].shape == torch.Size([1, out_channels[7], 2, 2])

    # Test EfficientNetV2 forward with 'm' arch
    out_channels = [24, 24, 48, 80, 160, 176, 304, 512, 1280]
    model = EfficientNetV2(arch='m', out_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 64, 64)
    feat = model(imgs)
    assert len(feat) == 9
    assert feat[0].shape == torch.Size([1, out_channels[0], 32, 32])
    assert feat[1].shape == torch.Size([1, out_channels[1], 32, 32])
    assert feat[2].shape == torch.Size([1, out_channels[2], 16, 16])
    assert feat[3].shape == torch.Size([1, out_channels[3], 8, 8])
    assert feat[4].shape == torch.Size([1, out_channels[4], 4, 4])
    assert feat[5].shape == torch.Size([1, out_channels[5], 4, 4])
    assert feat[6].shape == torch.Size([1, out_channels[6], 2, 2])
    assert feat[7].shape == torch.Size([1, out_channels[7], 2, 2])
    assert feat[8].shape == torch.Size([1, out_channels[8], 2, 2])

    # Test EfficientNetV2 forward with 'm' arch and GroupNorm
    out_channels = [24, 24, 48, 80, 160, 176, 304, 512, 1280]
    model = EfficientNetV2(
        arch='m',
        out_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 64, 64)
    feat = model(imgs)
    assert len(feat) == 9
    assert feat[0].shape == torch.Size([1, out_channels[0], 32, 32])
    assert feat[1].shape == torch.Size([1, out_channels[1], 32, 32])
    assert feat[2].shape == torch.Size([1, out_channels[2], 16, 16])
    assert feat[3].shape == torch.Size([1, out_channels[3], 8, 8])
    assert feat[4].shape == torch.Size([1, out_channels[4], 4, 4])
    assert feat[5].shape == torch.Size([1, out_channels[5], 4, 4])
    assert feat[6].shape == torch.Size([1, out_channels[6], 2, 2])
    assert feat[7].shape == torch.Size([1, out_channels[7], 2, 2])
    assert feat[8].shape == torch.Size([1, out_channels[8], 2, 2])

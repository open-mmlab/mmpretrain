# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import MobileNetV3
from mmpretrain.models.utils import InvertedResidual


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


def test_mobilenetv3_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = MobileNetV3()
        model.init_weights(pretrained=0)

    with pytest.raises(AssertionError):
        # arch must in [small, large]
        MobileNetV3(arch='others')

    with pytest.raises(ValueError):
        # frozen_stages must less than 13 when arch is small
        MobileNetV3(arch='small', frozen_stages=13)

    with pytest.raises(ValueError):
        # frozen_stages must less than 17 when arch is large
        MobileNetV3(arch='large', frozen_stages=17)

    with pytest.raises(ValueError):
        # max out_indices must less than 13 when arch is small
        MobileNetV3(arch='small', out_indices=(13, ))

    with pytest.raises(ValueError):
        # max out_indices must less than 17 when arch is large
        MobileNetV3(arch='large', out_indices=(17, ))

    # Test MobileNetV3
    model = MobileNetV3()
    model.init_weights()
    model.train()

    # Test MobileNetV3 with first stage frozen
    frozen_stages = 1
    model = MobileNetV3(frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    for i in range(0, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test MobileNetV3 with norm eval
    model = MobileNetV3(norm_eval=True, out_indices=range(0, 12))
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test MobileNetV3 forward with small arch
    model = MobileNetV3(out_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 13
    assert feat[0].shape == torch.Size([1, 16, 112, 112])
    assert feat[1].shape == torch.Size([1, 16, 56, 56])
    assert feat[2].shape == torch.Size([1, 24, 28, 28])
    assert feat[3].shape == torch.Size([1, 24, 28, 28])
    assert feat[4].shape == torch.Size([1, 40, 14, 14])
    assert feat[5].shape == torch.Size([1, 40, 14, 14])
    assert feat[6].shape == torch.Size([1, 40, 14, 14])
    assert feat[7].shape == torch.Size([1, 48, 14, 14])
    assert feat[8].shape == torch.Size([1, 48, 14, 14])
    assert feat[9].shape == torch.Size([1, 96, 7, 7])
    assert feat[10].shape == torch.Size([1, 96, 7, 7])
    assert feat[11].shape == torch.Size([1, 96, 7, 7])
    assert feat[12].shape == torch.Size([1, 576, 7, 7])

    # Test MobileNetV3 forward with small arch and GroupNorm
    model = MobileNetV3(
        out_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 13
    assert feat[0].shape == torch.Size([1, 16, 112, 112])
    assert feat[1].shape == torch.Size([1, 16, 56, 56])
    assert feat[2].shape == torch.Size([1, 24, 28, 28])
    assert feat[3].shape == torch.Size([1, 24, 28, 28])
    assert feat[4].shape == torch.Size([1, 40, 14, 14])
    assert feat[5].shape == torch.Size([1, 40, 14, 14])
    assert feat[6].shape == torch.Size([1, 40, 14, 14])
    assert feat[7].shape == torch.Size([1, 48, 14, 14])
    assert feat[8].shape == torch.Size([1, 48, 14, 14])
    assert feat[9].shape == torch.Size([1, 96, 7, 7])
    assert feat[10].shape == torch.Size([1, 96, 7, 7])
    assert feat[11].shape == torch.Size([1, 96, 7, 7])
    assert feat[12].shape == torch.Size([1, 576, 7, 7])

    # Test MobileNetV3 forward with large arch
    model = MobileNetV3(
        arch='large',
        out_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 17
    assert feat[0].shape == torch.Size([1, 16, 112, 112])
    assert feat[1].shape == torch.Size([1, 16, 112, 112])
    assert feat[2].shape == torch.Size([1, 24, 56, 56])
    assert feat[3].shape == torch.Size([1, 24, 56, 56])
    assert feat[4].shape == torch.Size([1, 40, 28, 28])
    assert feat[5].shape == torch.Size([1, 40, 28, 28])
    assert feat[6].shape == torch.Size([1, 40, 28, 28])
    assert feat[7].shape == torch.Size([1, 80, 14, 14])
    assert feat[8].shape == torch.Size([1, 80, 14, 14])
    assert feat[9].shape == torch.Size([1, 80, 14, 14])
    assert feat[10].shape == torch.Size([1, 80, 14, 14])
    assert feat[11].shape == torch.Size([1, 112, 14, 14])
    assert feat[12].shape == torch.Size([1, 112, 14, 14])
    assert feat[13].shape == torch.Size([1, 160, 7, 7])
    assert feat[14].shape == torch.Size([1, 160, 7, 7])
    assert feat[15].shape == torch.Size([1, 160, 7, 7])
    assert feat[16].shape == torch.Size([1, 960, 7, 7])

    # Test MobileNetV3 forward with large arch
    model = MobileNetV3(arch='large', out_indices=(0, ))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 16, 112, 112])

    # Test MobileNetV3 with checkpoint forward
    model = MobileNetV3(with_cp=True)
    for m in model.modules():
        if isinstance(m, InvertedResidual):
            assert m.with_cp
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 576, 7, 7])

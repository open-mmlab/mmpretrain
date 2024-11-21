# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import MobileNetV1

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

def test_mobilenetv1_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = MobileNetV1()
        model.init_weights(pretrained=0)

    with pytest.raises(ValueError):
        # frozen_stages must in range(-1, 8)
        MobileNetV1(frozen_stages=8)


    # Test MobileNetV2 with first stage frozen
    frozen_stages = 1
    model = MobileNetV1(frozen_stages=frozen_stages)
    model.init_weights()
    model.train()

    for mod in model.modules():
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
    model = MobileNetV1(norm_eval=True)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), False)

    
    # Test MobileNetV2 forward with dict(type='ReLU')
    model = MobileNetV1(act_cfg=dict(type='ReLU'))
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
    model = MobileNetV1()
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
    model = MobileNetV1(
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
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

    
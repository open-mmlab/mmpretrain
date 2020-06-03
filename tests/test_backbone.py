import pytest
import torch
import torch.nn as nn
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import MobileNetv2
from mmcls.models.backbones.mobilenet_v2 import InvertedResidual


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (InvertedResidual,)):
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
        InvertedResidual(64, 16,
                         stride=3, expand_ratio=6)

    # Test InvertedResidual with checkpoint forward, stride=1
    block = InvertedResidual(64, 16,
                             stride=1,
                             expand_ratio=6)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 16, 56, 56])

    # Test InvertedResidual with checkpoint forward, stride=2
    block = InvertedResidual(64, 16,
                             stride=2,
                             expand_ratio=6)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 16, 28, 28])

    # Test InvertedResidual with checkpoint forward
    block = InvertedResidual(64, 16,
                             stride=1,
                             expand_ratio=6,
                             with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 16, 56, 56])

    # Test InvertedResidual with activation=nn.ReLU
    block = InvertedResidual(64, 16,
                             stride=1,
                             expand_ratio=6,
                             activation=nn.ReLU)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 16, 56, 56])


def test_mobilenetv2_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = MobileNetv2()
        model.init_weights(pretrained=0)

    with pytest.raises(AssertionError):
        # frozen_stages must less than 7
        MobileNetv2(frozen_stages=8)

    # Test MobileNetv2
    model = MobileNetv2()
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test MobileNetv2 with first stage frozen
    frozen_stages = 1
    model = MobileNetv2(frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    assert model.bn1.training is False
    for layer in [model.conv1, model.bn1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test MobileNetv2 with first stage frozen
    model = MobileNetv2(bn_frozen=True)
    model.init_weights()
    model.train()
    assert model.bn1.training is False

    for i in range(1, 8):
        layer = getattr(model, f'layer{i}')

        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
                for params in mod.parameters():
                    params.requires_grad = False

    # Test MobileNetv2 forward with widen_factor=1.0
    model = MobileNetv2(widen_factor=1.0, activation=nn.ReLU6)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 8
    assert feat[0].shape == torch.Size([1, 16, 112, 112])
    assert feat[1].shape == torch.Size([1, 24, 56, 56])
    assert feat[2].shape == torch.Size([1, 32, 28, 28])
    assert feat[3].shape == torch.Size([1, 64, 14, 14])
    assert feat[4].shape == torch.Size([1, 96, 14, 14])
    assert feat[5].shape == torch.Size([1, 160, 7, 7])
    assert feat[6].shape == torch.Size([1, 320, 7, 7])

    # Test MobileNetv2 forward with activation=nn.ReLU
    model = MobileNetv2(widen_factor=1.0, activation=nn.ReLU)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 8
    assert feat[0].shape == torch.Size([1, 16, 112, 112])
    assert feat[1].shape == torch.Size([1, 24, 56, 56])
    assert feat[2].shape == torch.Size([1, 32, 28, 28])
    assert feat[3].shape == torch.Size([1, 64, 14, 14])
    assert feat[4].shape == torch.Size([1, 96, 14, 14])
    assert feat[5].shape == torch.Size([1, 160, 7, 7])
    assert feat[6].shape == torch.Size([1, 320, 7, 7])

    # Test MobileNetv2 with BatchNorm forward
    model = MobileNetv2(widen_factor=1.0, activation=nn.ReLU6)
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 8
    assert feat[0].shape == torch.Size([1, 16, 112, 112])
    assert feat[1].shape == torch.Size([1, 24, 56, 56])
    assert feat[2].shape == torch.Size([1, 32, 28, 28])
    assert feat[3].shape == torch.Size([1, 64, 14, 14])
    assert feat[4].shape == torch.Size([1, 96, 14, 14])
    assert feat[5].shape == torch.Size([1, 160, 7, 7])
    assert feat[6].shape == torch.Size([1, 320, 7, 7])

    # Test MobileNetv2 with BatchNorm forward
    model = MobileNetv2(widen_factor=1.0, activation=nn.ReLU6)
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 8
    assert feat[0].shape == torch.Size([1, 16, 112, 112])
    assert feat[1].shape == torch.Size([1, 24, 56, 56])
    assert feat[2].shape == torch.Size([1, 32, 28, 28])
    assert feat[3].shape == torch.Size([1, 64, 14, 14])
    assert feat[4].shape == torch.Size([1, 96, 14, 14])
    assert feat[5].shape == torch.Size([1, 160, 7, 7])
    assert feat[6].shape == torch.Size([1, 320, 7, 7])

    # Test MobileNetv2 with layers 1, 3, 5 out forward
    model = MobileNetv2(widen_factor=1.0, activation=nn.ReLU6,
                        out_indices=(0, 2, 4))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 16, 112, 112])
    assert feat[1].shape == torch.Size([1, 32, 28, 28])
    assert feat[2].shape == torch.Size([1, 96, 14, 14])

    # Test MobileNetv2 with checkpoint forward
    model = MobileNetv2(widen_factor=1.0, activation=nn.ReLU6,
                        with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 8
    assert feat[0].shape == torch.Size([1, 16, 112, 112])
    assert feat[1].shape == torch.Size([1, 24, 56, 56])
    assert feat[2].shape == torch.Size([1, 32, 28, 28])
    assert feat[3].shape == torch.Size([1, 64, 14, 14])
    assert feat[4].shape == torch.Size([1, 96, 14, 14])
    assert feat[5].shape == torch.Size([1, 160, 7, 7])
    assert feat[6].shape == torch.Size([1, 320, 7, 7])

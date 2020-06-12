import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import ShuffleNetv2
from mmcls.models.backbones.shufflenet_v2 import InvertedResidual


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


def test_shufflenetv2_invertedresidual():

    with pytest.raises(ValueError):
        # stride must be in [1, 2]
        InvertedResidual(24, 16, stride=3)

    with pytest.raises(AssertionError):
        # when stride==1, 16 == branch_features << 1
        InvertedResidual(24, 64, stride=1)

    # Test InvertedResidual forward
    block = InvertedResidual(24, 64, stride=2)
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 28, 28])

    # Test InvertedResidual with checkpoint forward
    block = InvertedResidual(24, 24, stride=1, with_cp=True)
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 24, 56, 56])


def test_ShuffleNetv2_backbone():

    with pytest.raises(ValueError):
        # groups must in 0.5, 1.0, 1.5, 2.0]
        ShuffleNetv2(widen_factor=3.0)

    # Test ShuffleNetv2 norm state
    model = ShuffleNetv2()
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test ShuffleNetv2 with first stage frozen
    frozen_stages = 1
    model = ShuffleNetv2(frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    for layer in [model.conv1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test ShuffleNetv2 with bn frozen
    model = ShuffleNetv2(bn_frozen=True)
    model.init_weights()
    model.train()

    for i in range(1, 4):
        layer = getattr(model, f'layer{i}')

        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
                for params in mod.parameters():
                    params.requires_grad = False

    # Test ShuffleNetv2 forward with widen_factor=1.0
    model = ShuffleNetv2(widen_factor=1.0)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 232, 28, 28])
    assert feat[1].shape == torch.Size([1, 464, 14, 14])
    assert feat[2].shape == torch.Size([1, 1024, 7, 7])
    assert feat[3].shape == torch.Size([1, 1024, 7, 7])

    # Test ShuffleNetv2 forward with layers 1 2 forward
    model = ShuffleNetv2(widen_factor=1.0, out_indices=(1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 464, 14, 14])
    assert feat[1].shape == torch.Size([1, 1024, 7, 7])

    # Test ShuffleNetv2 forward with checkpoint forward
    model = ShuffleNetv2(widen_factor=1.0, with_cp=True)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 232, 28, 28])
    assert feat[1].shape == torch.Size([1, 464, 14, 14])
    assert feat[2].shape == torch.Size([1, 1024, 7, 7])
    assert feat[3].shape == torch.Size([1, 1024, 7, 7])

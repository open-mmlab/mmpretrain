import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import ShuffleNetv1
from mmcls.models.backbones.shufflenet_v1 import ShuffleUnit


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (ShuffleUnit, )):
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


def test_shufflenetv1_shuffleuint():

    with pytest.raises(ValueError):
        # combine must be in ['add', 'concat']
        ShuffleUnit(24, 16, groups=3, first_block=True, combine='test')

    with pytest.raises(ValueError):
        # inplanes must be divisible by groups
        ShuffleUnit(64, 64, groups=3, first_block=True, combine='add')

    with pytest.raises(AssertionError):
        # inplanes must be equal tp = outplanes when combine='add'
        ShuffleUnit(64, 24, groups=3, first_block=True, combine='add')

    # Test ShuffleUnit with combine='add'
    block = ShuffleUnit(24, 24, groups=3, first_block=True, combine='add')
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 24, 56, 56])

    # Test ShuffleUnit with combine='concat'
    block = ShuffleUnit(24, 240, groups=3, first_block=True, combine='concat')
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 240, 28, 28])

    # Test ShuffleUnit with checkpoint forward
    block = ShuffleUnit(
        24, 24, groups=3, first_block=True, combine='add', with_cp=True)
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 24, 56, 56])


def test_shufflenetv1_backbone():

    with pytest.raises(ValueError):
        # frozen_stages must in range(-1, 4)
        ShuffleNetv1(frozen_stages=10)

    with pytest.raises(ValueError):
        # the item in out_indices must in range(0, 4)
        ShuffleNetv1(out_indices=[5])

    with pytest.raises(ValueError):
        # groups must in [1, 2, 3, 4, 8]
        ShuffleNetv1(groups=10)

    # Test ShuffleNetv1 norm state
    model = ShuffleNetv1()
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test ShuffleNetv1 with first stage frozen
    frozen_stages = 1
    model = ShuffleNetv1(frozen_stages=frozen_stages)
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

    # Test ShuffleNetv1 forward with groups=3
    model = ShuffleNetv1(groups=3)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 240, 28, 28])
    assert feat[1].shape == torch.Size([1, 480, 14, 14])
    assert feat[2].shape == torch.Size([1, 960, 7, 7])
    assert feat[3].shape == torch.Size([1, 960, 7, 7])

    # Test ShuffleNetv1 forward with GroupNorm forward
    model = ShuffleNetv1(
        groups=3, norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 240, 28, 28])
    assert feat[1].shape == torch.Size([1, 480, 14, 14])
    assert feat[2].shape == torch.Size([1, 960, 7, 7])
    assert feat[3].shape == torch.Size([1, 960, 7, 7])

    # Test ShuffleNetv1 forward with layers 1, 2 forward
    model = ShuffleNetv1(groups=3, out_indices=(1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 480, 14, 14])
    assert feat[1].shape == torch.Size([1, 960, 7, 7])
    assert feat[2].shape == torch.Size([1, 960, 7, 7])

    # Test ShuffleNetv1 forward with checkpoint forward
    model = ShuffleNetv1(groups=3, with_cp=True)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 240, 28, 28])
    assert feat[1].shape == torch.Size([1, 480, 14, 14])
    assert feat[2].shape == torch.Size([1, 960, 7, 7])
    assert feat[3].shape == torch.Size([1, 960, 7, 7])

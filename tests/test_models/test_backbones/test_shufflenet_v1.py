# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import ShuffleNetV1
from mmpretrain.models.backbones.shufflenet_v1 import ShuffleUnit


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

    with pytest.raises(AssertionError):
        # in_channels must be equal tp = outplanes when combine='add'
        ShuffleUnit(64, 24, groups=4, first_block=True, combine='add')

    # Test ShuffleUnit with combine='add'
    block = ShuffleUnit(24, 24, groups=3, first_block=True, combine='add')
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 24, 56, 56))

    # Test ShuffleUnit with combine='concat'
    block = ShuffleUnit(24, 240, groups=3, first_block=True, combine='concat')
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 240, 28, 28))

    # Test ShuffleUnit with checkpoint forward
    block = ShuffleUnit(
        24, 24, groups=3, first_block=True, combine='add', with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 24, 56, 56)
    x.requires_grad = True
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 24, 56, 56))


def test_shufflenetv1_backbone():

    with pytest.raises(ValueError):
        # frozen_stages must be in  range(-1, 4)
        ShuffleNetV1(frozen_stages=10)

    with pytest.raises(ValueError):
        # the item in out_indices must be in  range(0, 4)
        ShuffleNetV1(out_indices=[5])

    with pytest.raises(ValueError):
        # groups must be in  [1, 2, 3, 4, 8]
        ShuffleNetV1(groups=10)

    with pytest.raises(TypeError):
        # pretrained must be str or None
        model = ShuffleNetV1()
        model.init_weights(pretrained=1)

    # Test ShuffleNetV1 norm state
    model = ShuffleNetV1()
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    # Test ShuffleNetV1 with first stage frozen
    frozen_stages = 1
    model = ShuffleNetV1(frozen_stages=frozen_stages, out_indices=(0, 1, 2))
    model.init_weights()
    model.train()
    for param in model.conv1.parameters():
        assert param.requires_grad is False
    for i in range(frozen_stages):
        layer = model.layers[i]
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test ShuffleNetV1 forward with groups=1
    model = ShuffleNetV1(groups=1, out_indices=(0, 1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 144, 28, 28))
    assert feat[1].shape == torch.Size((1, 288, 14, 14))
    assert feat[2].shape == torch.Size((1, 576, 7, 7))

    # Test ShuffleNetV1 forward with groups=2
    model = ShuffleNetV1(groups=2, out_indices=(0, 1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 200, 28, 28))
    assert feat[1].shape == torch.Size((1, 400, 14, 14))
    assert feat[2].shape == torch.Size((1, 800, 7, 7))

    # Test ShuffleNetV1 forward with groups=3
    model = ShuffleNetV1(groups=3, out_indices=(0, 1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 240, 28, 28))
    assert feat[1].shape == torch.Size((1, 480, 14, 14))
    assert feat[2].shape == torch.Size((1, 960, 7, 7))

    # Test ShuffleNetV1 forward with groups=4
    model = ShuffleNetV1(groups=4, out_indices=(0, 1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 272, 28, 28))
    assert feat[1].shape == torch.Size((1, 544, 14, 14))
    assert feat[2].shape == torch.Size((1, 1088, 7, 7))

    # Test ShuffleNetV1 forward with groups=8
    model = ShuffleNetV1(groups=8, out_indices=(0, 1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 384, 28, 28))
    assert feat[1].shape == torch.Size((1, 768, 14, 14))
    assert feat[2].shape == torch.Size((1, 1536, 7, 7))

    # Test ShuffleNetV1 forward with GroupNorm forward
    model = ShuffleNetV1(
        groups=3,
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True),
        out_indices=(0, 1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 240, 28, 28))
    assert feat[1].shape == torch.Size((1, 480, 14, 14))
    assert feat[2].shape == torch.Size((1, 960, 7, 7))

    # Test ShuffleNetV1 forward with layers 1, 2 forward
    model = ShuffleNetV1(groups=3, out_indices=(1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 2
    assert feat[0].shape == torch.Size((1, 480, 14, 14))
    assert feat[1].shape == torch.Size((1, 960, 7, 7))

    # Test ShuffleNetV1 forward with layers 2 forward
    model = ShuffleNetV1(groups=3, out_indices=(2, ))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert isinstance(feat[0], torch.Tensor)
    assert feat[0].shape == torch.Size((1, 960, 7, 7))

    # Test ShuffleNetV1 forward with checkpoint forward
    model = ShuffleNetV1(groups=3, with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp

    # Test ShuffleNetV1 with norm_eval
    model = ShuffleNetV1(norm_eval=True)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), False)

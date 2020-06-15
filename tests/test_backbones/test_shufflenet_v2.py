import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import ShuffleNetv2
from mmcls.models.backbones.shufflenet_v2 import (InvertedResidual,
                                                  channel_shuffle,
                                                  make_divisible)


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


def test_channel_shuffle():
    x = torch.randn(1, 24, 56, 56)
    with pytest.raises(AssertionError):
        # num_channels should be divisible by groups
        channel_shuffle(x, 7)


def test_make_divisible():
    # test min_value is None
    make_divisible(34, 8, None)

    # test new_value < 0.9 * value
    make_divisible(10, 8, None)


def test_shufflenetv2_invertedresidual():

    with pytest.raises(AssertionError):
        # when stride==1, inplanes should be equal to planes // 2 * 2
        InvertedResidual(24, 32, stride=1)

    with pytest.raises(AssertionError):
        # when inplanes !=  planes // 2 * 2, stride should not be equal to 1.
        InvertedResidual(24, 32, stride=1)

    # Test InvertedResidual forward
    block = InvertedResidual(24, 48, stride=2)
    x = torch.randn(1, 24, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 48, 28, 28))

    # Test InvertedResidual with checkpoint forward
    block = InvertedResidual(48, 48, stride=1, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 48, 56, 56)
    x.requires_grad = True
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 48, 56, 56))


def test_shufflenetv2_backbone():

    with pytest.raises(ValueError):
        # groups must be in 0.5, 1.0, 1.5, 2.0]
        ShuffleNetv2(widen_factor=3.0)

    with pytest.raises(AssertionError):
        # frozen_stages must be in [0, 1, 2]
        ShuffleNetv2(widen_factor=3.0, frozen_stages=3)

    with pytest.raises(TypeError):
        # pretrained must be str or None
        model = ShuffleNetv2()
        model.init_weights(pretrained=1)

    # Test ShuffleNetv2 norm state
    model = ShuffleNetv2()
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    # Test ShuffleNetv2 with first stage frozen
    frozen_stages = 1
    model = ShuffleNetv2(frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    for param in model.conv1.parameters():
        assert param.requires_grad is False
    for i in range(0, frozen_stages):
        layer = model.layers[i]
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test ShuffleNetv2 with norm_eval
    model = ShuffleNetv2(norm_eval=True)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), False)

    # Test ShuffleNetv2 forward with widen_factor=0.5
    model = ShuffleNetv2(widen_factor=0.5)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 48, 28, 28))
    assert feat[1].shape == torch.Size((1, 96, 14, 14))
    assert feat[2].shape == torch.Size((1, 192, 7, 7))

    # Test ShuffleNetv2 forward with widen_factor=1.0
    model = ShuffleNetv2(widen_factor=1.0)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 116, 28, 28))
    assert feat[1].shape == torch.Size((1, 232, 14, 14))
    assert feat[2].shape == torch.Size((1, 464, 7, 7))

    # Test ShuffleNetv2 forward with widen_factor=1.5
    model = ShuffleNetv2(widen_factor=1.5)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 176, 28, 28))
    assert feat[1].shape == torch.Size((1, 352, 14, 14))
    assert feat[2].shape == torch.Size((1, 704, 7, 7))

    # Test ShuffleNetv2 forward with widen_factor=2.0
    model = ShuffleNetv2(widen_factor=2.0)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 244, 28, 28))
    assert feat[1].shape == torch.Size((1, 488, 14, 14))
    assert feat[2].shape == torch.Size((1, 976, 7, 7))

    # Test ShuffleNetv2 forward with layers 3 forward
    model = ShuffleNetv2(widen_factor=1.0, out_indices=(2, ))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert isinstance(feat, torch.Tensor)
    assert feat.shape == torch.Size((1, 464, 7, 7))

    # Test ShuffleNetv2 forward with layers 1 2 forward
    model = ShuffleNetv2(widen_factor=1.0, out_indices=(1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 2
    assert feat[0].shape == torch.Size((1, 232, 14, 14))
    assert feat[1].shape == torch.Size((1, 464, 7, 7))

    # Test ShuffleNetv2 forward with checkpoint forward
    model = ShuffleNetv2(widen_factor=1.0, with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp

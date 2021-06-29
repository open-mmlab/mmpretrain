import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import VGG
from mmcls.models.utils import (Augments, HybridEmbed, InvertedResidual,
                                PatchEmbed, SELayer, channel_shuffle,
                                make_divisible)


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def test_make_divisible():
    # test min_value is None
    result = make_divisible(34, 8, None)
    assert result == 32

    # test when new_value > min_ratio * value
    result = make_divisible(10, 8, min_ratio=0.9)
    assert result == 16

    # test min_value = 0.8
    result = make_divisible(33, 8, min_ratio=0.8)
    assert result == 32


def test_channel_shuffle():
    x = torch.randn(1, 24, 56, 56)
    with pytest.raises(AssertionError):
        # num_channels should be divisible by groups
        channel_shuffle(x, 7)

    groups = 3
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    out = channel_shuffle(x, groups)
    # test the output value when groups = 3
    for b in range(batch_size):
        for c in range(num_channels):
            c_out = c % channels_per_group * groups + c // channels_per_group
            for i in range(height):
                for j in range(width):
                    assert x[b, c, i, j] == out[b, c_out, i, j]


def test_inverted_residual():

    with pytest.raises(AssertionError):
        # stride must be in [1, 2]
        InvertedResidual(16, 16, 32, stride=3)

    with pytest.raises(AssertionError):
        # se_cfg must be None or dict
        InvertedResidual(16, 16, 32, se_cfg=list())

    # Add expand conv if in_channels and mid_channels is not the same
    assert InvertedResidual(32, 16, 32).with_expand_conv is False
    assert InvertedResidual(16, 16, 32).with_expand_conv is True

    # Test InvertedResidual forward, stride=1
    block = InvertedResidual(16, 16, 32, stride=1)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert getattr(block, 'se', None) is None
    assert block.with_res_shortcut
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward, stride=2
    block = InvertedResidual(16, 16, 32, stride=2)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert not block.with_res_shortcut
    assert x_out.shape == torch.Size((1, 16, 28, 28))

    # Test InvertedResidual forward with se layer
    se_cfg = dict(channels=32)
    block = InvertedResidual(16, 16, 32, stride=1, se_cfg=se_cfg)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert isinstance(block.se, SELayer)
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward without expand conv
    block = InvertedResidual(32, 16, 32)
    x = torch.randn(1, 32, 56, 56)
    x_out = block(x)
    assert getattr(block, 'expand_conv', None) is None
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward with GroupNorm
    block = InvertedResidual(
        16, 16, 32, norm_cfg=dict(type='GN', num_groups=2))
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    for m in block.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward with HSigmoid
    block = InvertedResidual(16, 16, 32, act_cfg=dict(type='HSigmoid'))
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward with checkpoint
    block = InvertedResidual(16, 16, 32, with_cp=True)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert block.with_cp
    assert x_out.shape == torch.Size((1, 16, 56, 56))


def test_augments():
    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4, ))

    # Test cutmix
    augments_cfg = dict(type='BatchCutMix', alpha=1., num_classes=10, prob=1.)
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))

    # Test mixup
    augments_cfg = dict(type='BatchMixup', alpha=1., num_classes=10, prob=1.)
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))

    # Test cutmixup
    augments_cfg = [
        dict(type='BatchCutMix', alpha=1., num_classes=10, prob=0.5),
        dict(type='BatchMixup', alpha=1., num_classes=10, prob=0.3)
    ]
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))

    augments_cfg = [
        dict(type='BatchCutMix', alpha=1., num_classes=10, prob=0.5),
        dict(type='BatchMixup', alpha=1., num_classes=10, prob=0.5)
    ]
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))

    augments_cfg = [
        dict(type='BatchCutMix', alpha=1., num_classes=10, prob=0.5),
        dict(type='BatchMixup', alpha=1., num_classes=10, prob=0.3),
        dict(type='Identity', num_classes=10, prob=0.2)
    ]
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))


def test_embed():

    # Test PatchEmbed
    patch_embed = PatchEmbed()
    img = torch.randn(1, 3, 224, 224)
    img = patch_embed(img)
    assert img.shape == torch.Size((1, 196, 768))

    # Test PatchEmbed with stride = 8
    conv_cfg = dict(kernel_size=16, stride=8)
    patch_embed = PatchEmbed(conv_cfg=conv_cfg)
    img = torch.randn(1, 3, 224, 224)
    img = patch_embed(img)
    assert img.shape == torch.Size((1, 729, 768))

    # Test VGG11 HybridEmbed
    backbone = VGG(11, norm_eval=True)
    backbone.init_weights()
    patch_embed = HybridEmbed(backbone)
    img = torch.randn(1, 3, 224, 224)
    img = patch_embed(img)
    assert img.shape == torch.Size((1, 49, 768))

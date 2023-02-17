# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import pytest
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import TIMMBackbone


def has_timm() -> bool:
    try:
        import timm  # noqa: F401
        return True
    except ImportError:
        return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


@unittest.skipIf(not has_timm(), 'timm is not installed')
def test_timm_backbone():
    """Test timm backbones, features_only=False (default)."""
    with pytest.raises(TypeError):
        # TIMMBackbone has 1 required positional argument: 'model_name'
        model = TIMMBackbone(pretrained=True)

    with pytest.raises(TypeError):
        # pretrained must be bool
        model = TIMMBackbone(model_name='resnet18', pretrained='model.pth')

    # Test resnet18 from timm
    model = TIMMBackbone(model_name='resnet18')
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)
    assert isinstance(model.timm_model.global_pool.pool, nn.Identity)
    assert isinstance(model.timm_model.fc, nn.Identity)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size((1, 512, 7, 7))

    # Test efficientnet_b1 with pretrained weights
    model = TIMMBackbone(model_name='efficientnet_b1', pretrained=True)
    model.init_weights()
    model.train()
    assert isinstance(model.timm_model.global_pool.pool, nn.Identity)
    assert isinstance(model.timm_model.classifier, nn.Identity)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size((1, 1280, 7, 7))

    # Test vit_tiny_patch16_224 with pretrained weights
    model = TIMMBackbone(model_name='vit_tiny_patch16_224', pretrained=True)
    model.init_weights()
    model.train()
    assert isinstance(model.timm_model.head, nn.Identity)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    # Disable the test since TIMM's behavior changes between 0.5.4 and 0.5.5
    # assert feat[0].shape == torch.Size((1, 197, 192))


@unittest.skipIf(not has_timm(), 'timm is not installed')
def test_timm_backbone_features_only():
    """Test timm backbones, features_only=True."""
    # Test different norm_layer, can be: 'SyncBN', 'BN2d', 'GN', 'LN', 'IN'
    # Test resnet18 from timm, norm_layer='BN2d'
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32,
        norm_layer='BN2d')

    # Test resnet18 from timm, norm_layer='SyncBN'
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32,
        norm_layer='SyncBN')

    # Test resnet18 from timm, output_stride=32
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 64, 112, 112))
    assert feats[1].shape == torch.Size((1, 64, 56, 56))
    assert feats[2].shape == torch.Size((1, 128, 28, 28))
    assert feats[3].shape == torch.Size((1, 256, 14, 14))
    assert feats[4].shape == torch.Size((1, 512, 7, 7))

    # Test resnet18 from timm, output_stride=32, out_indices=(1, 2, 3)
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32,
        out_indices=(1, 2, 3))
    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    assert len(feats) == 3
    assert feats[0].shape == torch.Size((1, 64, 56, 56))
    assert feats[1].shape == torch.Size((1, 128, 28, 28))
    assert feats[2].shape == torch.Size((1, 256, 14, 14))

    # Test resnet18 from timm, output_stride=16
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=16)
    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 64, 112, 112))
    assert feats[1].shape == torch.Size((1, 64, 56, 56))
    assert feats[2].shape == torch.Size((1, 128, 28, 28))
    assert feats[3].shape == torch.Size((1, 256, 14, 14))
    assert feats[4].shape == torch.Size((1, 512, 14, 14))

    # Test resnet18 from timm, output_stride=8
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 64, 112, 112))
    assert feats[1].shape == torch.Size((1, 64, 56, 56))
    assert feats[2].shape == torch.Size((1, 128, 28, 28))
    assert feats[3].shape == torch.Size((1, 256, 28, 28))
    assert feats[4].shape == torch.Size((1, 512, 28, 28))

    # Test efficientnet_b1 with pretrained weights
    model = TIMMBackbone(
        model_name='efficientnet_b1', features_only=True, pretrained=True)
    imgs = torch.randn(1, 3, 64, 64)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 16, 32, 32))
    assert feats[1].shape == torch.Size((1, 24, 16, 16))
    assert feats[2].shape == torch.Size((1, 40, 8, 8))
    assert feats[3].shape == torch.Size((1, 112, 4, 4))
    assert feats[4].shape == torch.Size((1, 320, 2, 2))

    # Test resnetv2_50x1_bitm from timm, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_50x1_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 8, 8)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 64, 4, 4))
    assert feats[1].shape == torch.Size((1, 256, 2, 2))
    assert feats[2].shape == torch.Size((1, 512, 1, 1))
    assert feats[3].shape == torch.Size((1, 1024, 1, 1))
    assert feats[4].shape == torch.Size((1, 2048, 1, 1))

    # Test resnetv2_50x3_bitm from timm, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_50x3_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 8, 8)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 192, 4, 4))
    assert feats[1].shape == torch.Size((1, 768, 2, 2))
    assert feats[2].shape == torch.Size((1, 1536, 1, 1))
    assert feats[3].shape == torch.Size((1, 3072, 1, 1))
    assert feats[4].shape == torch.Size((1, 6144, 1, 1))

    # Test resnetv2_101x1_bitm from timm, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_101x1_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 8, 8)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 64, 4, 4))
    assert feats[1].shape == torch.Size((1, 256, 2, 2))
    assert feats[2].shape == torch.Size((1, 512, 1, 1))
    assert feats[3].shape == torch.Size((1, 1024, 1, 1))
    assert feats[4].shape == torch.Size((1, 2048, 1, 1))

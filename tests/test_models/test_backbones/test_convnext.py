# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcls.models.backbones import ConvNeXt


def test_assertion():
    with pytest.raises(AssertionError):
        ConvNeXt(arch='unknown')

    with pytest.raises(AssertionError):
        # ConvNeXt arch dict should include 'embed_dims',
        ConvNeXt(arch=dict(channels=[2, 3, 4, 5]))

    with pytest.raises(AssertionError):
        # ConvNeXt arch dict should include 'embed_dims',
        ConvNeXt(arch=dict(depths=[2, 3, 4], channels=[2, 3, 4, 5]))


def test_convnext():

    # Test forward
    model = ConvNeXt(arch='tiny', out_indices=-1)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 768])

    # Test forward with multiple outputs
    model = ConvNeXt(arch='small', out_indices=(0, 1, 2, 3))

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 96])
    assert feat[1].shape == torch.Size([1, 192])
    assert feat[2].shape == torch.Size([1, 384])
    assert feat[3].shape == torch.Size([1, 768])

    # Test with custom arch
    model = ConvNeXt(
        arch={
            'depths': [2, 3, 4, 5, 6],
            'channels': [16, 32, 64, 128, 256]
        },
        out_indices=(0, 1, 2, 3, 4))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size([1, 16])
    assert feat[1].shape == torch.Size([1, 32])
    assert feat[2].shape == torch.Size([1, 64])
    assert feat[3].shape == torch.Size([1, 128])
    assert feat[4].shape == torch.Size([1, 256])

    # Test without gap before final norm
    model = ConvNeXt(
        arch='small', out_indices=(0, 1, 2, 3), gap_before_final_norm=False)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 96, 56, 56])
    assert feat[1].shape == torch.Size([1, 192, 28, 28])
    assert feat[2].shape == torch.Size([1, 384, 14, 14])
    assert feat[3].shape == torch.Size([1, 768, 7, 7])

    # Test frozen_stages
    model = ConvNeXt(arch='small', out_indices=(0, 1, 2, 3), frozen_stages=2)
    model.init_weights()
    model.train()

    for i in range(2):
        assert not model.downsample_layers[i].training
        assert not model.stages[i].training

    for i in range(2, 4):
        assert model.downsample_layers[i].training
        assert model.stages[i].training

    # Test Activation Checkpointing
    model = ConvNeXt(arch='tiny', out_indices=-1, with_cp=True)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 768])

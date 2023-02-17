# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmpretrain.models.backbones import EdgeNeXt


def test_assertion():
    with pytest.raises(AssertionError):
        EdgeNeXt(arch='unknown')

    with pytest.raises(AssertionError):
        # EdgeNeXt arch dict should include 'embed_dims',
        EdgeNeXt(arch=dict(channels=[24, 48, 88, 168]))

    with pytest.raises(AssertionError):
        # EdgeNeXt arch dict should include 'embed_dims',
        EdgeNeXt(arch=dict(depths=[2, 2, 6, 2], channels=[24, 48, 88, 168]))


def test_edgenext():

    # Test forward
    model = EdgeNeXt(arch='xxsmall', out_indices=-1)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 168])

    # Test forward with multiple outputs
    model = EdgeNeXt(arch='xxsmall', out_indices=(0, 1, 2, 3))
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 24])
    assert feat[1].shape == torch.Size([1, 48])
    assert feat[2].shape == torch.Size([1, 88])
    assert feat[3].shape == torch.Size([1, 168])

    # Test with custom arch
    model = EdgeNeXt(
        arch={
            'depths': [2, 3, 4, 5],
            'channels': [20, 40, 80, 160],
            'num_heads': [4, 4, 4, 4]
        },
        out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 20])
    assert feat[1].shape == torch.Size([1, 40])
    assert feat[2].shape == torch.Size([1, 80])
    assert feat[3].shape == torch.Size([1, 160])

    # Test without gap before final norm
    model = EdgeNeXt(
        arch='small', out_indices=(0, 1, 2, 3), gap_before_final_norm=False)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 48, 56, 56])
    assert feat[1].shape == torch.Size([1, 96, 28, 28])
    assert feat[2].shape == torch.Size([1, 160, 14, 14])
    assert feat[3].shape == torch.Size([1, 304, 7, 7])

    # Test frozen_stages
    model = EdgeNeXt(arch='small', out_indices=(0, 1, 2, 3), frozen_stages=2)
    model.init_weights()
    model.train()

    for i in range(2):
        assert not model.downsample_layers[i].training
        assert not model.stages[i].training

    for i in range(2, 4):
        assert model.downsample_layers[i].training
        assert model.stages[i].training

# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmpretrain.models.backbones import ConvMixer


def test_assertion():
    with pytest.raises(AssertionError):
        ConvMixer(arch='unknown')

    with pytest.raises(AssertionError):
        # ConvMixer arch dict should include essential_keys,
        ConvMixer(arch=dict(channels=[2, 3, 4, 5]))

    with pytest.raises(AssertionError):
        # ConvMixer out_indices should be valid depth.
        ConvMixer(out_indices=-100)


@torch.no_grad()  # To save memory
def test_convmixer():

    # Test forward
    model = ConvMixer(arch='768/32')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 768, 32, 32])

    # Test forward with multiple outputs
    model = ConvMixer(arch='768/32', out_indices=range(32))

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 32
    for f in feat:
        assert f.shape == torch.Size([1, 768, 32, 32])

    # Test with custom arch
    model = ConvMixer(
        arch={
            'embed_dims': 99,
            'depth': 5,
            'patch_size': 5,
            'kernel_size': 9
        },
        out_indices=range(5))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    for f in feat:
        assert f.shape == torch.Size([1, 99, 44, 44])

    # Test with even kernel size arch
    model = ConvMixer(arch={
        'embed_dims': 99,
        'depth': 5,
        'patch_size': 5,
        'kernel_size': 8
    })
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 99, 44, 44])

    # Test frozen_stages
    model = ConvMixer(arch='768/32', frozen_stages=10)
    model.init_weights()
    model.train()

    for i in range(10):
        assert not model.stages[i].training

    for i in range(10, 32):
        assert model.stages[i].training

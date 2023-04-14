# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmpretrain.models.backbones import MobileViT


def test_assertion():
    with pytest.raises(AssertionError):
        MobileViT(arch='unknown')

    with pytest.raises(AssertionError):
        # MobileViT out_indices should be valid depth.
        MobileViT(out_indices=-100)


def test_mobilevit():

    # Test forward
    model = MobileViT(arch='small')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 640, 8, 8])

    # Test custom arch
    model = MobileViT(arch=[
        ['mobilenetv2', 16, 1, 1, 2],
        ['mobilenetv2', 24, 2, 3, 2],
        ['mobilevit', 48, 2, 64, 128, 2, 2],
        ['mobilevit', 64, 2, 80, 160, 4, 2],
        ['mobilevit', 80, 2, 96, 192, 3, 2],
    ])
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 320, 8, 8])

    # Test last_exp_factor
    model = MobileViT(arch='small', last_exp_factor=8)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 1280, 8, 8])

    # Test stem_channels
    model = MobileViT(arch='small', stem_channels=32)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 640, 8, 8])

    # Test forward with multiple outputs
    model = MobileViT(arch='small', out_indices=range(5))

    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size([1, 32, 128, 128])
    assert feat[1].shape == torch.Size([1, 64, 64, 64])
    assert feat[2].shape == torch.Size([1, 96, 32, 32])
    assert feat[3].shape == torch.Size([1, 128, 16, 16])
    assert feat[4].shape == torch.Size([1, 640, 8, 8])

    # Test frozen_stages
    model = MobileViT(arch='small', frozen_stages=2)
    model.init_weights()
    model.train()

    for i in range(2):
        assert not model.layers[i].training

    for i in range(2, 5):
        assert model.layers[i].training

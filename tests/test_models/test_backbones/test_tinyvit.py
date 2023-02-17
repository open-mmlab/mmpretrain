# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmpretrain.models.backbones import TinyViT


def test_assertion():
    with pytest.raises(AssertionError):
        TinyViT(arch='unknown')

    with pytest.raises(AssertionError):
        # MobileViT out_indices should be valid depth.
        TinyViT(out_indices=-100)


def test_tinyvit():

    # Test forward
    model = TinyViT(arch='5m')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 320])

    # Test forward with multiple outputs
    model = TinyViT(arch='5m', out_indices=(0, 1, 2, 3))
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 128])
    assert feat[1].shape == torch.Size([1, 160])
    assert feat[2].shape == torch.Size([1, 320])
    assert feat[3].shape == torch.Size([1, 320])

    # Test with custom arch
    model = TinyViT(
        arch={
            'depths': [2, 3, 4, 5],
            'channels': [64, 128, 256, 448],
            'num_heads': [4, 4, 4, 4]
        },
        out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 128])
    assert feat[1].shape == torch.Size([1, 256])
    assert feat[2].shape == torch.Size([1, 448])
    assert feat[3].shape == torch.Size([1, 448])

    # Test without gap before final norm
    model = TinyViT(
        arch='21m', out_indices=(0, 1, 2, 3), gap_before_final_norm=False)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4

    assert feat[0].shape == torch.Size([1, 192, 28, 28])
    assert feat[1].shape == torch.Size([1, 384, 14, 14])
    assert feat[2].shape == torch.Size([1, 576, 7, 7])
    assert feat[3].shape == torch.Size([1, 576, 7, 7])

    # Test frozen_stages
    model = TinyViT(arch='11m', out_indices=(0, 1, 2, 3), frozen_stages=2)
    model.init_weights()
    model.train()

    for i in range(2):
        assert not model.stages[i].training

    for i in range(2, 4):
        assert model.stages[i].training

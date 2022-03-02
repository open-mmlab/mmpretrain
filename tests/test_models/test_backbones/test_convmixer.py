# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcls.models.backbones import ConvMixer


def test_assertion():
    with pytest.raises(AssertionError):
        ConvMixer(arch='unknown')

    with pytest.raises(AssertionError):
        # ConvMixer arch dict should include essential_keys,
        ConvMixer(arch=dict(channels=[2, 3, 4, 5]))


def test_convmixer():

    # Test forward
    model = ConvMixer(arch='768/32')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 768])

    # Test with custom arch
    model = ConvMixer(arch={
        'embed_dims': 999,
        'depth': 5,
        'patch_size': 5,
        'kernel_size': 9
    })
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 999])

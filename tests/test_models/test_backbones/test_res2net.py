# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpretrain.models.backbones import Res2Net


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_resnet_cifar():
    # Only support depth 50, 101 and 152
    with pytest.raises(KeyError):
        Res2Net(depth=18)

    # test the feature map size when depth is 50
    # and deep_stem=True, avg_down=True
    model = Res2Net(
        depth=50, out_indices=(0, 1, 2, 3), deep_stem=True, avg_down=True)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model.stem(imgs)
    assert feat.shape == (1, 64, 112, 112)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, 256, 56, 56)
    assert feat[1].shape == (1, 512, 28, 28)
    assert feat[2].shape == (1, 1024, 14, 14)
    assert feat[3].shape == (1, 2048, 7, 7)

    # test the feature map size when depth is 101
    # and deep_stem=False, avg_down=False
    model = Res2Net(
        depth=101, out_indices=(0, 1, 2, 3), deep_stem=False, avg_down=False)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model.conv1(imgs)
    assert feat.shape == (1, 64, 112, 112)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, 256, 56, 56)
    assert feat[1].shape == (1, 512, 28, 28)
    assert feat[2].shape == (1, 1024, 14, 14)
    assert feat[3].shape == (1, 2048, 7, 7)

    # Test Res2Net with first stage frozen
    frozen_stages = 1
    model = Res2Net(depth=50, frozen_stages=frozen_stages, deep_stem=False)
    model.init_weights()
    model.train()
    assert check_norm_state([model.norm1], False)
    for param in model.conv1.parameters():
        assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

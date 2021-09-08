# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import TIMMBackbone


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_timm_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = TIMMBackbone()
        model.init_weights(pretrained=0)

    # Test resnet18 from timm
    model = TIMMBackbone(model_name='resnet18')
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size((1, 512, 7, 7))

    # Test efficientnet_b1 with pretrained weights
    model = TIMMBackbone(model_name='efficientnet_b1', pretrained=True)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size((1, 1280, 7, 7))

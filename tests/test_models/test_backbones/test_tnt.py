# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import TNT


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_tnt_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = TNT()
        model.init_weights(pretrained=0)

    # Test tnt_base_patch16_224
    model = TNT()
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size((1, 640))

    # Test tnt with embed_dims=768
    arch = {
        'embed_dims_outer': 768,
        'embed_dims_inner': 48,
        'num_layers': 12,
        'num_heads_outer': 6,
        'num_heads_inner': 4
    }
    model = TNT(arch=arch)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size((1, 768))

# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import MlpMixer


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_mlp_mixer_backbone():
    cfg_ori = dict(
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ])

    with pytest.raises(AssertionError):
        # test invalid arch
        cfg = deepcopy(cfg_ori)
        cfg['arch'] = 'unknown'
        MlpMixer(**cfg)

    with pytest.raises(AssertionError):
        # test arch without essential keys
        cfg = deepcopy(cfg_ori)
        cfg['arch'] = {
            'num_layers': 24,
            'tokens_mlp_dims': 384,
            'channels_mlp_dims': 3072,
        }
        MlpMixer(**cfg)

    # Test MlpMixer base model with input size of 224
    # and patch size of 16
    model = MlpMixer(**cfg_ori)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(3, 3, 224, 224)
    feat = model(imgs)[-1]
    assert feat.shape == (3, 768, 196)

    # Test MlpMixer with multi out indices
    cfg = deepcopy(cfg_ori)
    cfg['out_indices'] = [-3, -2, -1]
    model = MlpMixer(**cfg)
    for out in model(imgs):
        assert out.shape == (3, 768, 196)

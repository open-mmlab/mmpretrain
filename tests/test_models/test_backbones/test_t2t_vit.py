# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import T2T_ViT


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


def test_vit_backbone():

    cfg_ori = dict(
        img_size=224,
        in_channels=3,
        embed_dims=384,
        t2t_cfg=dict(
            token_dims=64,
            use_performer=False,
        ),
        num_layers=14,
        layer_cfgs=dict(
            num_heads=6,
            feedforward_channels=3 * 384,  # mlp_ratio = 3
        ),
        drop_path_rate=0.1,
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=.02),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
        ])

    with pytest.raises(NotImplementedError):
        # test if use performer
        cfg = deepcopy(cfg_ori)
        cfg['t2t_cfg']['use_performer'] = True
        T2T_ViT(**cfg)

    # Test T2T-ViT model with input size of 224
    model = T2T_ViT(**cfg_ori)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(3, 3, 224, 224)
    patch_token, cls_token = model(imgs)[-1]
    assert cls_token.shape == (3, 384)
    assert patch_token.shape == (3, 384, 14, 14)

    # Test custom arch T2T-ViT without output cls token
    cfg = deepcopy(cfg_ori)
    cfg['embed_dims'] = 256
    cfg['num_layers'] = 16
    cfg['layer_cfgs'] = dict(num_heads=8, feedforward_channels=1024)
    cfg['output_cls_token'] = False

    model = T2T_ViT(**cfg)
    patch_token = model(imgs)[-1]
    assert patch_token.shape == (3, 256, 14, 14)

    # Test T2T_ViT with multi out indices
    cfg = deepcopy(cfg_ori)
    cfg['out_indices'] = [-3, -2, -1]
    model = T2T_ViT(**cfg)
    for out in model(imgs):
        assert out[0].shape == (3, 384, 14, 14)
        assert out[1].shape == (3, 384)

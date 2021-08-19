# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import VisionTransformer


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
        VisionTransformer(**cfg)

    with pytest.raises(AssertionError):
        # test arch without essential keys
        cfg = deepcopy(cfg_ori)
        cfg['arch'] = {
            'num_layers': 24,
            'num_heads': 16,
            'feedforward_channels': 4096
        }
        VisionTransformer(**cfg)

    # Test ViT base model with input size of 224
    # and patch size of 16
    model = VisionTransformer(**cfg_ori)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(3, 3, 224, 224)
    patch_token, cls_token = model(imgs)[-1]
    assert cls_token.shape == (3, 768)
    assert patch_token.shape == (3, 768, 14, 14)

    # Test custom arch ViT without output cls token
    cfg = deepcopy(cfg_ori)
    cfg['arch'] = {
        'embed_dims': 128,
        'num_layers': 24,
        'num_heads': 16,
        'feedforward_channels': 1024
    }
    cfg['output_cls_token'] = False
    model = VisionTransformer(**cfg)
    patch_token = model(imgs)[-1]
    assert patch_token.shape == (3, 128, 14, 14)

    # Test ViT with multi out indices
    cfg = deepcopy(cfg_ori)
    cfg['out_indices'] = [-3, -2, -1]
    model = VisionTransformer(**cfg)
    for out in model(imgs):
        assert out[0].shape == (3, 768, 14, 14)
        assert out[1].shape == (3, 768)


def timm_resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Timm version pos embed resize function.
    # Refers to https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py # noqa:E501
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0,
                                                                 num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old,
                                      -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3,
                                      1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def test_vit_weight_init():
    # test weight init cfg
    pretrain_cfg = dict(
        arch='b',
        img_size=224,
        patch_size=16,
        init_cfg=[dict(type='Constant', val=1., layer='Conv2d')])
    pretrain_model = VisionTransformer(**pretrain_cfg)
    pretrain_model.init_weights()
    assert torch.allclose(pretrain_model.patch_embed.projection.weight,
                          torch.tensor(1.))
    assert pretrain_model.pos_embed.abs().sum() > 0

    pos_embed_weight = pretrain_model.pos_embed.detach()
    tmpdir = tempfile.gettempdir()
    checkpoint = os.path.join(tmpdir, 'test.pth')
    torch.save(pretrain_model.state_dict(), checkpoint)

    # test load checkpoint
    finetune_cfg = dict(
        arch='b',
        img_size=224,
        patch_size=16,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint))
    finetune_model = VisionTransformer(**finetune_cfg)
    finetune_model.init_weights()
    assert torch.allclose(finetune_model.pos_embed, pos_embed_weight)

    # test load checkpoint with different img_size
    finetune_cfg = dict(
        arch='b',
        img_size=384,
        patch_size=16,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint))
    finetune_model = VisionTransformer(**finetune_cfg)
    finetune_model.init_weights()
    resized_pos_embed = timm_resize_pos_embed(pos_embed_weight,
                                              finetune_model.pos_embed)
    assert torch.allclose(finetune_model.pos_embed, resized_pos_embed)

    os.remove(checkpoint)

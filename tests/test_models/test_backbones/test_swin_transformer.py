# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from math import ceil

import numpy as np
import pytest
import torch
from mmcv.runner import load_checkpoint, save_checkpoint

from mmcls.models.backbones import SwinTransformer


def test_assertion():
    """Test Swin Transformer backbone."""
    with pytest.raises(AssertionError):
        # Swin Transformer arch string should be in
        SwinTransformer(arch='unknown')

    with pytest.raises(AssertionError):
        # Swin Transformer arch dict should include 'embed_dims',
        # 'depths' and 'num_head' keys.
        SwinTransformer(arch=dict(embed_dims=96, depths=[2, 2, 18, 2]))


def test_forward():
    # Test tiny arch forward
    model = SwinTransformer(arch='Tiny')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    output = model(imgs)
    assert len(output) == 1
    assert output[0].shape == (1, 768, 7, 7)

    # Test small arch forward
    model = SwinTransformer(arch='small')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    output = model(imgs)
    assert len(output) == 1
    assert output[0].shape == (1, 768, 7, 7)

    # Test base arch forward
    model = SwinTransformer(arch='B')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    output = model(imgs)
    assert len(output) == 1
    assert output[0].shape == (1, 1024, 7, 7)

    # Test large arch forward
    model = SwinTransformer(arch='l')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    output = model(imgs)
    assert len(output) == 1
    assert output[0].shape == (1, 1536, 7, 7)

    # Test base arch with window_size=12, image_size=384
    model = SwinTransformer(
        arch='base',
        img_size=384,
        stage_cfgs=dict(block_cfgs=dict(window_size=12)))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 384, 384)
    output = model(imgs)
    assert len(output) == 1
    assert output[0].shape == (1, 1024, 12, 12)


def test_structure():
    # Test small with use_abs_pos_embed = True
    model = SwinTransformer(arch='small', use_abs_pos_embed=True)
    assert model.absolute_pos_embed.shape == (1, 3136, 96)

    # Test small with use_abs_pos_embed = False
    model = SwinTransformer(arch='small', use_abs_pos_embed=False)
    assert not hasattr(model, 'absolute_pos_embed')

    # Test small with auto_pad = True
    model = SwinTransformer(
        arch='small',
        auto_pad=True,
        stage_cfgs=dict(
            block_cfgs={'window_size': 7},
            downsample_cfg={
                'kernel_size': (3, 2),
            }))

    # stage 1
    input_h = int(224 / 4 / 3)
    expect_h = ceil(input_h / 7) * 7
    input_w = int(224 / 4 / 2)
    expect_w = ceil(input_w / 7) * 7
    assert model.stages[1].blocks[0].attn.pad_b == expect_h - input_h
    assert model.stages[1].blocks[0].attn.pad_r == expect_w - input_w

    # stage 2
    input_h = int(224 / 4 / 3 / 3)
    # input_h is smaller than window_size, shrink the window_size to input_h.
    expect_h = input_h
    input_w = int(224 / 4 / 2 / 2)
    expect_w = ceil(input_w / input_h) * input_h
    assert model.stages[2].blocks[0].attn.pad_b == expect_h - input_h
    assert model.stages[2].blocks[0].attn.pad_r == expect_w - input_w

    # stage 3
    input_h = int(224 / 4 / 3 / 3 / 3)
    expect_h = input_h
    input_w = int(224 / 4 / 2 / 2 / 2)
    expect_w = ceil(input_w / input_h) * input_h
    assert model.stages[3].blocks[0].attn.pad_b == expect_h - input_h
    assert model.stages[3].blocks[0].attn.pad_r == expect_w - input_w

    # Test small with auto_pad = False
    with pytest.raises(AssertionError):
        model = SwinTransformer(
            arch='small',
            auto_pad=False,
            stage_cfgs=dict(
                block_cfgs={'window_size': 7},
                downsample_cfg={
                    'kernel_size': (3, 2),
                }))

    # Test drop_path_rate decay
    model = SwinTransformer(
        arch='small',
        drop_path_rate=0.2,
    )
    depths = model.arch_settings['depths']
    pos = 0
    for i, depth in enumerate(depths):
        for j in range(depth):
            block = model.stages[i].blocks[j]
            expect_prob = 0.2 / (sum(depths) - 1) * pos
            assert np.isclose(block.ffn.dropout_layer.drop_prob, expect_prob)
            assert np.isclose(block.attn.drop.drop_prob, expect_prob)
            pos += 1


def test_load_checkpoint():
    model = SwinTransformer(arch='tiny')
    ckpt_path = os.path.join(tempfile.gettempdir(), 'ckpt.pth')

    assert model._version == 2

    # test load v2 checkpoint
    save_checkpoint(model, ckpt_path)
    load_checkpoint(model, ckpt_path, strict=True)

    # test load v1 checkpoint
    setattr(model, 'norm', model.norm3)
    model._version = 1
    del model.norm3
    save_checkpoint(model, ckpt_path)
    model = SwinTransformer(arch='tiny')
    load_checkpoint(model, ckpt_path, strict=True)

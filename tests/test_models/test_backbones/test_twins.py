# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcls.models.backbones.twins import (PCPVT, SVT,
                                          GlobalSubsampledAttention,
                                          LocallyGroupedSelfAttention)


def test_LSA_module():
    lsa = LocallyGroupedSelfAttention(embed_dims=32, window_size=3)
    outs = lsa(torch.randn(1, 3136, 32), (56, 56))
    assert outs.shape == torch.Size([1, 3136, 32])


def test_GSA_module():
    gsa = GlobalSubsampledAttention(embed_dims=32, num_heads=8)
    outs = gsa(torch.randn(1, 3136, 32), (56, 56))
    assert outs.shape == torch.Size([1, 3136, 32])


def test_pcpvt():
    # Test feature map output
    H, W = (64, 64)
    temp = torch.randn((1, 3, H, W))
    pcpvt_cfg = dict(
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        norm_after_stage=True,
        final_norm=False)

    model = PCPVT(**pcpvt_cfg)
    model.init_weights()
    outs = model(temp)
    assert len(outs) == 4
    assert outs[0].shape == (1, 32, H // 4, W // 4)
    assert outs[1].shape == (1, 64, H // 8, W // 8)
    assert outs[2].shape == (1, 160, H // 16, W // 16)
    assert outs[3].shape == (1, 256, H // 32, W // 32)

    pcpvt_cfg.update(
        dict(out_indices=(3, ), norm_after_stage=False, final_norm=False))
    model = PCPVT(**pcpvt_cfg)
    model.init_weights()
    outs = model(temp)
    assert len(outs) == 1
    assert outs[-1].shape == (1, 256, H // 32, W // 32)


def test_svt():
    # Test feature map output
    H, W = (64, 64)
    temp = torch.randn((1, 3, H, W))
    svt_cfg = dict(
        out_indices=(
            0,
            1,
            2,
        ),
        embed_dims=[
            32,
            64,
            128,
        ],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        depths=[4, 4, 4],
        windiow_sizes=[7, 7, 7],
        norm_after_stage=True,
        final_norm=False)

    model = SVT(**svt_cfg)
    model.init_weights()
    outs = model(temp)
    assert len(outs) == 3
    assert outs[0].shape == (1, 32, H // 4, W // 4)
    assert outs[1].shape == (1, 64, H // 8, W // 8)
    assert outs[2].shape == (1, 128, H // 16, W // 16)

    svt_cfg.update(
        dict(out_indices=(2, ), norm_after_stage=False, final_norm=True))
    model = SVT(**svt_cfg)
    model.init_weights()
    outs = model(temp)
    assert len(outs) == 1
    assert outs[-1].shape == (1, 128, H // 16, W // 16)


def test_svt_init():
    path = 'PATH_THAT_DO_NOT_EXIST'
    # Test all combinations of pretrained and init_cfg
    # pretrained=None, init_cfg=None
    model = SVT(pretrained=None, init_cfg=None)
    assert model.init_cfg is None
    model.init_weights()

    # pretrained=None
    # init_cfg loads pretrain from an non-existent file
    model = SVT(
        pretrained=None, init_cfg=dict(type='Pretrained', checkpoint=path))
    assert model.init_cfg == dict(
        type='Pretrained',
        checkpoint=path), (model.init_cfg,
                           dict(type='Pretrained', checkpoint=path))
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained=None
    # init_cfg=123, whose type is unsupported
    model = SVT(pretrained=None, init_cfg=123)
    with pytest.raises(TypeError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg=None
    model = SVT(pretrained=path, init_cfg=None)
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = SVT(
            pretrained=path, init_cfg=dict(type='Pretrained', checkpoint=path))
    with pytest.raises(AssertionError):
        model = SVT(pretrained=path, init_cfg=123)

    # pretrain=123, whose type is unsupported
    # init_cfg=None
    with pytest.raises(TypeError):
        model = SVT(pretrained=123, init_cfg=None)

    # pretrain=123, whose type is unsupported
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = SVT(
            pretrained=123, init_cfg=dict(type='Pretrained', checkpoint=path))

    # pretrain=123, whose type is unsupported
    # init_cfg=123, whose type is unsupported
    with pytest.raises(AssertionError):
        model = SVT(pretrained=123, init_cfg=123)


def test_pcpvt_init():
    path = 'PATH_THAT_DO_NOT_EXIST'
    # Test all combinations of pretrained and init_cfg
    # pretrained=None, init_cfg=None
    model = PCPVT(pretrained=None, init_cfg=None)
    assert model.init_cfg is None
    model.init_weights()

    # pretrained=None
    # init_cfg loads pretrain from an non-existent file
    model = PCPVT(
        pretrained=None, init_cfg=dict(type='Pretrained', checkpoint=path))
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained=None
    # init_cfg=123, whose type is unsupported
    model = PCPVT(pretrained=None, init_cfg=123)
    with pytest.raises(TypeError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg=None
    model = PCPVT(pretrained=path, init_cfg=None)
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = PCPVT(
            pretrained=path, init_cfg=dict(type='Pretrained', checkpoint=path))
    with pytest.raises(AssertionError):
        model = PCPVT(pretrained=path, init_cfg=123)

    # pretrain=123, whose type is unsupported
    # init_cfg=None
    with pytest.raises(TypeError):
        model = PCPVT(pretrained=123, init_cfg=None)

    # pretrain=123, whose type is unsupported
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = PCPVT(
            pretrained=123, init_cfg=dict(type='Pretrained', checkpoint=path))

    # pretrain=123, whose type is unsupported
    # init_cfg=123, whose type is unsupported
    with pytest.raises(AssertionError):
        model = PCPVT(pretrained=123, init_cfg=123)

# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch
import torch.nn as nn

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
    # test init
    path = 'PATH_THAT_DO_NOT_EXIST'

    # init_cfg loads pretrain from an non-existent file
    model = PCPVT('s', init_cfg=dict(type='Pretrained', checkpoint=path))
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)

    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # init_cfg=123, whose type is unsupported
    model = PCPVT('s', init_cfg=123)
    with pytest.raises(TypeError):
        model.init_weights()

    H, W = (64, 64)
    temp = torch.randn((1, 3, H, W))

    # test output last feat
    model = PCPVT('small')
    model.init_weights()
    outs = model(temp)
    assert len(outs) == 1
    assert outs[-1].shape == (1, 512, H // 32, W // 32)

    # test with mutil outputs
    model = PCPVT('small', out_indices=(0, 1, 2, 3))
    model.init_weights()
    outs = model(temp)
    assert len(outs) == 4
    assert outs[0].shape == (1, 64, H // 4, W // 4)
    assert outs[1].shape == (1, 128, H // 8, W // 8)
    assert outs[2].shape == (1, 320, H // 16, W // 16)
    assert outs[3].shape == (1, 512, H // 32, W // 32)

    # test with arch of dict
    arch = {
        'embed_dims': [64, 128, 320, 512],
        'depths': [3, 4, 18, 3],
        'num_heads': [1, 2, 5, 8],
        'patch_sizes': [4, 2, 2, 2],
        'strides': [4, 2, 2, 2],
        'mlp_ratios': [8, 8, 4, 4],
        'sr_ratios': [8, 4, 2, 1]
    }

    pcpvt_arch = copy.deepcopy(arch)
    model = PCPVT(pcpvt_arch, out_indices=(0, 1, 2, 3))
    model.init_weights()
    outs = model(temp)
    assert len(outs) == 4
    assert outs[0].shape == (1, 64, H // 4, W // 4)
    assert outs[1].shape == (1, 128, H // 8, W // 8)
    assert outs[2].shape == (1, 320, H // 16, W // 16)
    assert outs[3].shape == (1, 512, H // 32, W // 32)

    # assert length of arch value not equal
    pcpvt_arch = copy.deepcopy(arch)
    pcpvt_arch['sr_ratios'] = [8, 4, 2]
    with pytest.raises(AssertionError):
        model = PCPVT(pcpvt_arch, out_indices=(0, 1, 2, 3))

    # assert lack arch essential_keys
    pcpvt_arch = copy.deepcopy(arch)
    del pcpvt_arch['sr_ratios']
    with pytest.raises(AssertionError):
        model = PCPVT(pcpvt_arch, out_indices=(0, 1, 2, 3))

    # assert arch value not list
    pcpvt_arch = copy.deepcopy(arch)
    pcpvt_arch['sr_ratios'] = 1
    with pytest.raises(AssertionError):
        model = PCPVT(pcpvt_arch, out_indices=(0, 1, 2, 3))

    pcpvt_arch = copy.deepcopy(arch)
    pcpvt_arch['sr_ratios'] = '1, 2, 3, 4'
    with pytest.raises(AssertionError):
        model = PCPVT(pcpvt_arch, out_indices=(0, 1, 2, 3))

    # test norm_after_stage is bool True
    model = PCPVT('small', norm_after_stage=True, norm_cfg=dict(type='LN'))
    for i in range(model.num_stage):
        assert hasattr(model, f'norm_after_stage{i}')
        assert isinstance(getattr(model, f'norm_after_stage{i}'), nn.LayerNorm)

    # test norm_after_stage is bool Flase
    model = PCPVT('small', norm_after_stage=False)
    for i in range(model.num_stage):
        assert hasattr(model, f'norm_after_stage{i}')
        assert isinstance(getattr(model, f'norm_after_stage{i}'), nn.Identity)

    # test norm_after_stage is bool list
    norm_after_stage = [False, True, False, True]
    model = PCPVT('small', norm_after_stage=norm_after_stage)
    assert len(norm_after_stage) == model.num_stage
    for i in range(model.num_stage):
        assert hasattr(model, f'norm_after_stage{i}')
        norm_layer = getattr(model, f'norm_after_stage{i}')
        if norm_after_stage[i]:
            assert isinstance(norm_layer, nn.LayerNorm)
        else:
            assert isinstance(norm_layer, nn.Identity)

    # test norm_after_stage is not bool list
    norm_after_stage = [False, 'True', False, True]
    with pytest.raises(AssertionError):
        model = PCPVT('small', norm_after_stage=norm_after_stage)


def test_svt():
    # test init
    path = 'PATH_THAT_DO_NOT_EXIST'

    # init_cfg loads pretrain from an non-existent file
    model = SVT('s', init_cfg=dict(type='Pretrained', checkpoint=path))
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)

    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # init_cfg=123, whose type is unsupported
    model = SVT('s', init_cfg=123)
    with pytest.raises(TypeError):
        model.init_weights()

    # Test feature map output
    H, W = (64, 64)
    temp = torch.randn((1, 3, H, W))

    model = SVT('s')
    model.init_weights()
    outs = model(temp)
    assert len(outs) == 1
    assert outs[-1].shape == (1, 512, H // 32, W // 32)

    # test with mutil outputs
    model = SVT('small', out_indices=(0, 1, 2, 3))
    model.init_weights()
    outs = model(temp)
    assert len(outs) == 4
    assert outs[0].shape == (1, 64, H // 4, W // 4)
    assert outs[1].shape == (1, 128, H // 8, W // 8)
    assert outs[2].shape == (1, 256, H // 16, W // 16)
    assert outs[3].shape == (1, 512, H // 32, W // 32)

    # test with arch of dict
    arch = {
        'embed_dims': [96, 192, 384, 768],
        'depths': [2, 2, 18, 2],
        'num_heads': [3, 6, 12, 24],
        'patch_sizes': [4, 2, 2, 2],
        'strides': [4, 2, 2, 2],
        'mlp_ratios': [4, 4, 4, 4],
        'sr_ratios': [8, 4, 2, 1],
        'window_sizes': [7, 7, 7, 7]
    }
    model = SVT(arch, out_indices=(0, 1, 2, 3))
    model.init_weights()
    outs = model(temp)
    assert len(outs) == 4
    assert outs[0].shape == (1, 96, H // 4, W // 4)
    assert outs[1].shape == (1, 192, H // 8, W // 8)
    assert outs[2].shape == (1, 384, H // 16, W // 16)
    assert outs[3].shape == (1, 768, H // 32, W // 32)

    # assert length of arch value not equal
    svt_arch = copy.deepcopy(arch)
    svt_arch['sr_ratios'] = [8, 4, 2]
    with pytest.raises(AssertionError):
        model = SVT(svt_arch, out_indices=(0, 1, 2, 3))

    # assert lack arch essential_keys
    svt_arch = copy.deepcopy(arch)
    del svt_arch['window_sizes']
    with pytest.raises(AssertionError):
        model = SVT(svt_arch, out_indices=(0, 1, 2, 3))

    # assert arch value not list
    svt_arch = copy.deepcopy(arch)
    svt_arch['sr_ratios'] = 1
    with pytest.raises(AssertionError):
        model = SVT(svt_arch, out_indices=(0, 1, 2, 3))

    svt_arch = copy.deepcopy(arch)
    svt_arch['sr_ratios'] = '1, 2, 3, 4'
    with pytest.raises(AssertionError):
        model = SVT(svt_arch, out_indices=(0, 1, 2, 3))

    # test norm_after_stage is bool True
    model = SVT('small', norm_after_stage=True, norm_cfg=dict(type='LN'))
    for i in range(model.num_stage):
        assert hasattr(model, f'norm_after_stage{i}')
        assert isinstance(getattr(model, f'norm_after_stage{i}'), nn.LayerNorm)

    # test norm_after_stage is bool Flase
    model = SVT('small', norm_after_stage=False)
    for i in range(model.num_stage):
        assert hasattr(model, f'norm_after_stage{i}')
        assert isinstance(getattr(model, f'norm_after_stage{i}'), nn.Identity)

    # test norm_after_stage is bool list
    norm_after_stage = [False, True, False, True]
    model = SVT('small', norm_after_stage=norm_after_stage)
    assert len(norm_after_stage) == model.num_stage
    for i in range(model.num_stage):
        assert hasattr(model, f'norm_after_stage{i}')
        norm_layer = getattr(model, f'norm_after_stage{i}')
        if norm_after_stage[i]:
            assert isinstance(norm_layer, nn.LayerNorm)
        else:
            assert isinstance(norm_layer, nn.Identity)

    # test norm_after_stage is not bool list
    norm_after_stage = [False, 'True', False, True]
    with pytest.raises(AssertionError):
        model = SVT('small', norm_after_stage=norm_after_stage)

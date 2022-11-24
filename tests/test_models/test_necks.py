# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcls.models.necks import (GeneralizedMeanPooling, GlobalAveragePooling,
                                HRFuseScales, LinearReduction,
                                TransformerTokenMergeNeck)


def test_gap_neck():

    # test 1d gap_neck
    neck = GlobalAveragePooling(dim=1)
    # batch_size, num_features, feature_size
    fake_input = torch.rand(1, 16, 24)

    output = neck(fake_input)
    # batch_size, num_features
    assert output.shape == (1, 16)

    # test 1d gap_neck
    neck = GlobalAveragePooling(dim=2)
    # batch_size, num_features, feature_size(2)
    fake_input = torch.rand(1, 16, 24, 24)

    output = neck(fake_input)
    # batch_size, num_features
    assert output.shape == (1, 16)

    # test 1d gap_neck
    neck = GlobalAveragePooling(dim=3)
    # batch_size, num_features, feature_size(3)
    fake_input = torch.rand(1, 16, 24, 24, 5)

    output = neck(fake_input)
    # batch_size, num_features
    assert output.shape == (1, 16)

    with pytest.raises(AssertionError):
        # dim must in [1, 2, 3]
        GlobalAveragePooling(dim='other')


def test_gem_neck():

    # test gem_neck
    neck = GeneralizedMeanPooling()
    # batch_size, num_features, feature_size(2)
    fake_input = torch.rand(1, 16, 24, 24)

    output = neck(fake_input)
    # batch_size, num_features
    assert output.shape == (1, 16)

    # test tuple input gem_neck
    neck = GeneralizedMeanPooling()
    # batch_size, num_features, feature_size(2)
    fake_input = (torch.rand(1, 8, 24, 24), torch.rand(1, 16, 24, 24))

    output = neck(fake_input)
    # batch_size, num_features
    assert output[0].shape == (1, 8)
    assert output[1].shape == (1, 16)

    with pytest.raises(AssertionError):
        # p must be a value greater then 1
        GeneralizedMeanPooling(p=0.5)


def test_hr_fuse_scales():

    in_channels = (18, 32, 64, 128)
    neck = HRFuseScales(in_channels=in_channels, out_channels=1024)

    feat_size = 56
    inputs = []
    for in_channel in in_channels:
        input_tensor = torch.rand(3, in_channel, feat_size, feat_size)
        inputs.append(input_tensor)
        feat_size = feat_size // 2

    with pytest.raises(AssertionError):
        neck(inputs)

    outs = neck(tuple(inputs))
    assert isinstance(outs, tuple)
    assert len(outs) == 1
    assert outs[0].shape == (3, 1024, 7, 7)


def test_linear_reduction():
    # test linear_reduction without `act_cfg` and `norm_cfg`
    neck = LinearReduction(10, 5, None, None)
    neck.eval()
    assert isinstance(neck.act, torch.nn.Identity)
    assert isinstance(neck.norm, torch.nn.Identity)

    # batch_size, in_channels, out_channels
    fake_input = torch.rand(1, 10)
    output = neck(fake_input)
    # batch_size, out_features
    assert output[-1].shape == (1, 5)

    # batch_size, in_features, feature_size(2)
    fake_input = (torch.rand(1, 20), torch.rand(1, 10))

    output = neck(fake_input)
    # batch_size, out_features
    assert output[-1].shape == (1, 5)

    # test linear_reduction with `init_cfg`
    neck = LinearReduction(
        10, 5, init_cfg=dict(type='Xavier', layer=['Linear']))

    # test linear_reduction with `act_cfg` and `norm_cfg`
    neck = LinearReduction(
        10, 5, act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BN1d'))
    neck.eval()

    assert isinstance(neck.act, torch.nn.ReLU)
    assert isinstance(neck.norm, torch.nn.BatchNorm1d)

    # batch_size, in_channels, out_channels
    fake_input = torch.rand(1, 10)
    output = neck(fake_input)
    # batch_size, out_features
    assert output[-1].shape == (1, 5)
    #
    # # batch_size, in_features, feature_size(2)
    fake_input = (torch.rand(1, 20), torch.rand(1, 10))

    output = neck(fake_input)
    # batch_size, out_features
    assert output[-1].shape == (1, 5)

    with pytest.raises(AssertionError):
        neck([])


def test_transformer_token_merge_neck():
    # test invalid input
    with pytest.raises(AssertionError):
        neck = TransformerTokenMergeNeck(mode='xx')

    neck = TransformerTokenMergeNeck(mode='concat')

    # test fake patch_token and cls_token
    fake_input = [[torch.rand(1, 8, 14, 14), torch.rand(1, 8)]]
    output = neck(fake_input)
    assert output[0].shape == (1, 16)

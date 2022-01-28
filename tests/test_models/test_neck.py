# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcls.models.necks import GeneralizedMeanPooling, GlobalAveragePooling


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

# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcls.models.necks import GlobalAveragePooling


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

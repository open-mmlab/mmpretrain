import pytest
import torch

from mmcls.models.necks import GlobalAveragePooling


def test_gap_neck():

    # test 1d gap_neck
    neck = GlobalAveragePooling(mode='1d')
    # batch_size, num_features, feature_size
    fake_input = torch.rand(1, 16, 24)

    output = neck(fake_input)
    # batch_size, num_features
    assert output.shape == (1, 16)

    # test 1d gap_neck
    neck = GlobalAveragePooling(mode='2d')
    # batch_size, num_features, feature_size(2)
    fake_input = torch.rand(1, 16, 24, 24)

    output = neck(fake_input)
    # batch_size, num_features
    assert output.shape == (1, 16)

    # test 1d gap_neck
    neck = GlobalAveragePooling(mode='3d')
    # batch_size, num_features, feature_size(3)
    fake_input = torch.rand(1, 16, 24, 24, 5)

    output = neck(fake_input)
    # batch_size, num_features
    assert output.shape == (1, 16)

    with pytest.raises(NotImplementedError):
        # mode must in ['1d', '2d', '3d']
        GlobalAveragePooling(mode='other')

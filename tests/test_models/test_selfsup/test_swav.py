# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import SwAV
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_swav():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'to_rgb': True
    }
    backbone = dict(
        type='ResNet',
        depth=18,
        norm_cfg=dict(type='BN'),
        zero_init_residual=True)
    neck = dict(
        type='SwAVNeck',
        in_channels=512,
        hid_channels=2,
        out_channels=2,
        norm_cfg=dict(type='BN1d'),
        with_avg_pool=True)
    head = dict(
        type='SwAVHead',
        loss=dict(
            type='SwAVLoss',
            feat_dim=2,  # equal to neck['out_channels']
            epsilon=0.05,
            temperature=0.1,
            num_crops=[2, 6]))

    alg = SwAV(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=data_preprocessor)

    fake_data = {
        'inputs': [
            torch.randn((2, 3, 224, 224)),
            torch.randn((2, 3, 224, 224)),
            torch.randn((2, 3, 96, 96)),
            torch.randn((2, 3, 96, 96)),
            torch.randn((2, 3, 96, 96)),
            torch.randn((2, 3, 96, 96)),
            torch.randn((2, 3, 96, 96)),
            torch.randn((2, 3, 96, 96))
        ],
        'data_samples': [DataSample() for _ in range(2)]
    }

    fake_inputs = alg.data_preprocessor(fake_data)
    fake_outputs = alg(**fake_inputs, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

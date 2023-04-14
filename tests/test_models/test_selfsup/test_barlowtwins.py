# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import BarlowTwins
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_barlowtwins():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'to_rgb': True
    }
    backbone = dict(type='ResNet', depth=18, norm_cfg=dict(type='BN'))
    neck = dict(
        type='NonLinearNeck',
        in_channels=512,
        hid_channels=2,
        out_channels=2,
        num_layers=3,
        with_last_bn=False,
        with_last_bn_affine=False,
        with_avg_pool=True,
        norm_cfg=dict(type='BN1d'))
    head = dict(
        type='LatentCrossCorrelationHead',
        in_channels=2,
        loss=dict(type='CrossCorrelationLoss'))

    alg = BarlowTwins(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=data_preprocessor)

    fake_data = {
        'inputs':
        [torch.randn((2, 3, 224, 224)),
         torch.randn((2, 3, 224, 224))],
        'data_sample': [DataSample() for _ in range(2)]
    }

    fake_inputs = alg.data_preprocessor(fake_data)
    fake_loss = alg(**fake_inputs, mode='loss')
    assert isinstance(fake_loss['loss'].item(), float)

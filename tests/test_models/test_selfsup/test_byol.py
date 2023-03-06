# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import BYOL
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_byol():
    data_preprocessor = dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        to_rgb=True)
    backbone = dict(type='ResNet', depth=18, norm_cfg=dict(type='BN'))
    neck = dict(
        type='NonLinearNeck',
        in_channels=512,
        hid_channels=2,
        out_channels=2,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True,
        norm_cfg=dict(type='BN1d'))
    head = dict(
        type='LatentPredictHead',
        loss=dict(type='CosineSimilarityLoss'),
        predictor=dict(
            type='NonLinearNeck',
            in_channels=2,
            hid_channels=2,
            out_channels=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False,
            norm_cfg=dict(type='BN1d')))

    alg = BYOL(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=data_preprocessor)

    fake_data = {
        'inputs':
        [torch.randn((2, 3, 224, 224)),
         torch.randn((2, 3, 224, 224))],
        'data_samples': [DataSample() for _ in range(2)]
    }
    fake_inputs = alg.data_preprocessor(fake_data)

    fake_loss = alg(**fake_inputs, mode='loss')
    assert isinstance(fake_loss['loss'].item(), float)
    assert fake_loss['loss'].item() > -4

    fake_feats = alg(fake_inputs['inputs'][0], mode='tensor')
    assert list(fake_feats[0].shape) == [2, 512, 7, 7]

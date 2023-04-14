# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch

from mmpretrain.models import SimSiam
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_simsiam():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'to_rgb': True,
    }
    backbone = dict(
        type='ResNet',
        depth=18,
        norm_cfg=dict(type='BN'),
        zero_init_residual=True)
    neck = dict(
        type='NonLinearNeck',
        in_channels=512,
        hid_channels=2,
        out_channels=2,
        num_layers=3,
        with_last_bn_affine=False,
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
            with_avg_pool=False,
            with_last_bn=False,
            with_last_bias=True,
            norm_cfg=dict(type='BN1d')))

    alg = SimSiam(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=copy.deepcopy(data_preprocessor))

    fake_data = {
        'inputs':
        [torch.randn((2, 3, 224, 224)),
         torch.randn((2, 3, 224, 224))],
        'data_samples': [DataSample() for _ in range(2)]
    }
    fake_inputs = alg.data_preprocessor(fake_data)
    fake_loss = alg(**fake_inputs, mode='loss')
    assert fake_loss['loss'] > -1

    # test extract
    fake_feat = alg(fake_inputs['inputs'][0], mode='tensor')
    assert fake_feat[0].size() == torch.Size([2, 512, 7, 7])

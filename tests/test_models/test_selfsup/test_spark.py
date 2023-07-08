# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import SparK
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_spark():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'to_rgb': True
    }

    backbone = dict(
        type='SparseResNet',
        depth=50,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.05,
        norm_cfg=dict(type='BN'))
    neck = dict(
        type='SparKLightDecoder',
        feature_dim=512,
        upsample_ratio=32,  # equal to downsample_raito
        mid_channels=0,
        norm_cfg=dict(type='BN'),
        last_act=False)
    head = dict(
        type='SparKPretrainHead',
        loss=dict(type='PixelReconstructionLoss', criterion='L2'))

    alg = SparK(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=data_preprocessor,
        enc_dec_norm_cfg=dict(type='BN'),
    )

    fake_data = {
        'inputs': torch.randn((2, 3, 224, 224)),
        'data_sample': [DataSample() for _ in range(2)]
    }

    fake_inputs = alg.data_preprocessor(fake_data)
    fake_loss = alg(**fake_inputs, mode='loss')
    assert isinstance(fake_loss['loss'].item(), float)

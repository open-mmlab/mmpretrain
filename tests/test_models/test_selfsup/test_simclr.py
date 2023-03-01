# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import SimCLR
from mmpretrain.structures import DataSample

backbone = dict(type='ResNet', depth=18, norm_cfg=dict(type='BN'))
neck = dict(
    type='NonLinearNeck',  # SimCLR non-linear neck
    in_channels=512,
    hid_channels=2,
    out_channels=2,
    num_layers=2,
    with_avg_pool=True,
    norm_cfg=dict(type='BN1d'))
head = dict(
    type='ContrastiveHead',
    loss=dict(type='CrossEntropyLoss'),
    temperature=0.1)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_simclr():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'to_rgb': True,
    }

    alg = SimCLR(
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

    # test extract
    fake_feat = alg(fake_inputs['inputs'][0], mode='tensor')
    assert fake_feat[0].size() == torch.Size([2, 512, 7, 7])

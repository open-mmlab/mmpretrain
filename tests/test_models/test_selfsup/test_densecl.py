# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import DenseCL
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_densecl():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'to_rgb': True
    }
    queue_len = 32
    feat_dim = 2
    momentum = 0.001
    loss_lambda = 0.5
    backbone = dict(type='ResNet', depth=18, norm_cfg=dict(type='BN'))
    neck = dict(
        type='DenseCLNeck',
        in_channels=512,
        hid_channels=2,
        out_channels=2,
        num_grid=None)
    head = dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2)

    alg = DenseCL(
        backbone=backbone,
        neck=neck,
        head=head,
        queue_len=queue_len,
        feat_dim=feat_dim,
        momentum=momentum,
        loss_lambda=loss_lambda,
        data_preprocessor=data_preprocessor)

    # test init
    assert alg.queue.size() == torch.Size([feat_dim, queue_len])
    assert alg.queue2.size() == torch.Size([feat_dim, queue_len])

    # test loss
    fake_data = {
        'inputs':
        [torch.randn((2, 3, 224, 224)),
         torch.randn((2, 3, 224, 224))],
        'data_samples': [DataSample() for _ in range(2)]
    }
    fake_inputs = alg.data_preprocessor(fake_data)
    fake_loss = alg(**fake_inputs, mode='loss')
    assert isinstance(fake_loss['loss_single'].item(), float)
    assert isinstance(fake_loss['loss_dense'].item(), float)
    assert fake_loss['loss_single'].item() > 0
    assert fake_loss['loss_dense'].item() > 0
    assert alg.queue_ptr.item() == 2
    assert alg.queue2_ptr.item() == 2

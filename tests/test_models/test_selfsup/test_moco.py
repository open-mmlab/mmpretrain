# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import MoCo
from mmpretrain.structures import DataSample

queue_len = 32
feat_dim = 2
momentum = 0.001
backbone = dict(type='ResNet', depth=18, norm_cfg=dict(type='BN'))
neck = dict(
    type='MoCoV2Neck',
    in_channels=512,
    hid_channels=2,
    out_channels=2,
    with_avg_pool=True)
head = dict(
    type='ContrastiveHead',
    loss=dict(type='CrossEntropyLoss'),
    temperature=0.2)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_moco():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'to_rgb': True
    }

    alg = MoCo(
        backbone=backbone,
        neck=neck,
        head=head,
        queue_len=queue_len,
        feat_dim=feat_dim,
        momentum=momentum,
        data_preprocessor=data_preprocessor)
    assert alg.queue.size() == torch.Size([feat_dim, queue_len])

    fake_data = {
        'inputs':
        [torch.randn((2, 3, 224, 224)),
         torch.randn((2, 3, 224, 224))],
        'data_samples': [DataSample() for _ in range(2)]
    }

    fake_inputs = alg.data_preprocessor(fake_data)
    fake_loss = alg(**fake_inputs, mode='loss')
    assert fake_loss['loss'] > 0
    assert alg.queue_ptr.item() == 2

    # test extract
    fake_feats = alg(fake_inputs['inputs'][0], mode='tensor')
    assert fake_feats[0].size() == torch.Size([2, 512, 7, 7])

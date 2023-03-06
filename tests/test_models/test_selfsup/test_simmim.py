# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import SimMIM, SimMIMSwinTransformer
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_simmim_swin():
    backbone = dict(
        arch='B',
        img_size=192,
        stage_cfgs=dict(block_cfgs=dict(window_size=6)))
    simmim_backbone = SimMIMSwinTransformer(**backbone)
    simmim_backbone.init_weights()
    fake_inputs = torch.randn((2, 3, 192, 192))
    fake_mask = torch.rand((2, 48, 48))

    # test with mask
    fake_outputs = simmim_backbone(fake_inputs, fake_mask)[0]
    assert fake_outputs.shape == torch.Size([2, 1024, 6, 6])

    # test without mask
    fake_outputs = simmim_backbone(fake_inputs, None)
    assert len(fake_outputs) == 1
    assert fake_outputs[0].shape == torch.Size([2, 1024, 6, 6])


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_simmim():
    data_preprocessor = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_rgb': True
    }

    # model config
    backbone = dict(
        type='SimMIMSwinTransformer',
        arch='B',
        img_size=192,
        stage_cfgs=dict(block_cfgs=dict(window_size=6)))
    neck = dict(
        type='SimMIMLinearDecoder', in_channels=128 * 2**3, encoder_stride=32)
    head = dict(
        type='SimMIMHead',
        patch_size=4,
        loss=dict(type='PixelReconstructionLoss', criterion='L1', channel=3))

    model = SimMIM(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=data_preprocessor)

    # test forward_train
    fake_data_sample = DataSample()
    fake_mask = torch.rand((48, 48))
    fake_data_sample.set_mask(fake_mask)
    fake_data = {
        'inputs': torch.randn((2, 3, 192, 192)),
        'data_samples': [fake_data_sample for _ in range(2)]
    }

    fake_inputs = model.data_preprocessor(fake_data)
    fake_outputs = model(**fake_inputs, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

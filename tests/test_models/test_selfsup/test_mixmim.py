# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import MixMIM, MixMIMPretrainTransformer
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mixmmim_backbone():
    mixmmim_backbone = MixMIMPretrainTransformer(
        arch=dict(embed_dims=128, depths=[2, 2, 4, 2], num_heads=[4, 4, 4, 4]))
    mixmmim_backbone.init_weights()
    fake_inputs = torch.randn((1, 3, 224, 224))

    # test with mask
    fake_outputs, fake_mask_s4 = mixmmim_backbone(fake_inputs)
    assert fake_outputs.shape == torch.Size([1, 49, 1024])
    assert fake_mask_s4.shape == torch.Size([1, 49, 1])

    # test without mask
    fake_outputs = mixmmim_backbone(fake_inputs, None)
    assert len(fake_outputs) == 1
    assert fake_outputs[0].shape == torch.Size([1, 1024])


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_simmim():
    data_preprocessor = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_rgb': True
    }

    # model config
    backbone = dict(
        type='MixMIMPretrainTransformer',
        arch='B',
        drop_rate=0.0,
        drop_path_rate=0.0)
    neck = dict(
        type='MixMIMPretrainDecoder',
        num_patches=49,
        encoder_stride=32,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16)
    head = dict(
        type='MixMIMPretrainHead',
        norm_pix=True,
        loss=dict(type='PixelReconstructionLoss', criterion='L2'))

    model = MixMIM(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=data_preprocessor)

    # test forward_train
    fake_data_sample = DataSample()
    fake_data = {
        'inputs': torch.randn((2, 3, 224, 224)),
        'data_samples': [fake_data_sample for _ in range(2)]
    }

    fake_inputs = model.data_preprocessor(fake_data)
    fake_outputs = model(**fake_inputs, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import CAEPretrainViT, CAE
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_cae_vit():
    backbone = dict(arch='b', patch_size=16, layer_scale_init_value=0.1)

    cae_backbone = CAEPretrainViT(**backbone)
    cae_backbone.init_weights()
    fake_inputs = torch.randn((2, 3, 224, 224))
    fake_mask = torch.zeros((2, 196)).bool()
    fake_mask[:, 75:150] = 1

    # test with mask
    fake_outputs = cae_backbone(fake_inputs, fake_mask)
    assert list(fake_outputs.shape) == [2, 122, 768]

    # test without mask
    fake_outputs = cae_backbone(fake_inputs, None)
    assert len(fake_outputs[0]) == 2
    assert fake_outputs[0][0].shape == torch.Size([2, 768, 14, 14])
    assert fake_outputs[0][1].shape == torch.Size([2, 768])


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_cae():
    data_preprocessor = dict(
        type='TwoNormDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        second_mean=[-31.875, -31.875, -31.875],
        second_std=[318.75, 318.75, 318.75],
        to_rgb=True)

    # model settings
    backbone = dict(
        type='CAEPretrainViT',
        arch='b',
        patch_size=16,
        layer_scale_init_value=0.1)
    neck = dict(
        type='CAENeck',
        embed_dims=768,
        num_heads=12,
        regressor_depth=4,
        decoder_depth=4,
        mlp_ratio=4,
        layer_scale_init_value=0.1)
    head = dict(type='CAEHead', loss=dict(type='CAELoss', lambd=2))
    target_generator = dict(type='DALL-E')

    model = CAE(
        backbone=backbone,
        neck=neck,
        head=head,
        target_generator=target_generator,
        data_preprocessor=data_preprocessor)

    fake_img = torch.rand((1, 3, 224, 224))
    fake_target_img = torch.rand((1, 3, 112, 112))
    fake_mask = torch.zeros((196)).bool()
    fake_mask[75:150] = 1
    fake_data_sample = DataSample()
    fake_data_sample.set_mask(fake_mask)
    fake_data = {
        'inputs': [fake_img, fake_target_img],
        'data_samples': [fake_data_sample]
    }

    fake_inputs = model.data_preprocessor(fake_data)
    fake_outputs = model(**fake_inputs, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

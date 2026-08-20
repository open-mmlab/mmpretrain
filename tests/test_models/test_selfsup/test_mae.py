# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import MAE, MAEHiViT, MAEViT
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mae_vit():
    backbone = dict(arch='b', patch_size=16, mask_ratio=0.75)
    mae_backbone = MAEViT(**backbone)
    mae_backbone.init_weights()
    fake_inputs = torch.randn((2, 3, 224, 224))

    # test with mask
    fake_outputs = mae_backbone(fake_inputs)[0]
    assert list(fake_outputs.shape) == [2, 50, 768]

    # test without mask
    outputs_without_mask = mae_backbone(fake_inputs, None)
    outputs_with_mask_disabled = mae_backbone(fake_inputs, False)
    assert outputs_without_mask[0].shape == torch.Size([2, 197, 768])
    assert len(outputs_with_mask_disabled) == 1
    assert torch.allclose(outputs_with_mask_disabled[0],
                          outputs_without_mask[0])


def test_mae_hivit_without_mask():
    backbone = MAEHiViT(
        arch=dict(embed_dims=32, depths=[1, 1, 1], num_heads=4),
        img_size=32,
        patch_size=8,
        inner_patches=2,
        out_indices=[0])
    fake_inputs = torch.randn((1, 3, 32, 32))

    outputs_without_mask = backbone(fake_inputs, None)
    outputs_with_mask_disabled = backbone(fake_inputs, False)
    assert len(outputs_with_mask_disabled) == len(outputs_without_mask)
    for actual, expected in zip(outputs_with_mask_disabled,
                                outputs_without_mask):
        assert torch.allclose(actual, expected)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mae():
    data_preprocessor = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_rgb': True
    }
    backbone = dict(type='MAEViT', arch='b', patch_size=16, mask_ratio=0.75)
    neck = dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    )
    loss = dict(type='PixelReconstructionLoss', criterion='L2')
    head = dict(
        type='MAEPretrainHead', norm_pix=False, patch_size=16, loss=loss)

    alg = MAE(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=data_preprocessor)

    fake_data = {
        'inputs': torch.randn((2, 3, 224, 224)),
        'data_samples': [DataSample() for _ in range(2)]
    }
    fake_inputs = alg.data_preprocessor(fake_data)
    fake_outputs = alg(**fake_inputs, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

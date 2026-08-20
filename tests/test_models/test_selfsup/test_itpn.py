# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import iTPN, iTPNHiViT
from mmpretrain.structures import DataSample


@pytest.mark.parametrize('reconstruction_type', ['pixel', 'clip'])
def test_itpn_backbone_without_mask(reconstruction_type):
    backbone = iTPNHiViT(
        arch=dict(embed_dims=32, depths=[1, 1, 1], num_heads=4),
        img_size=32,
        patch_size=8,
        inner_patches=2,
        out_indices=[0],
        reconstruction_type=reconstruction_type)
    fake_inputs = torch.randn((1, 3, 32, 32))

    outputs_without_mask = backbone(fake_inputs, None)
    outputs_with_mask_disabled = backbone(fake_inputs, False)
    assert len(outputs_with_mask_disabled) == len(outputs_without_mask)
    for actual, expected in zip(outputs_with_mask_disabled,
                                outputs_without_mask):
        assert torch.allclose(actual, expected)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_itpn():
    data_preprocessor = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_rgb': True
    }
    backbone = dict(
        type='iTPNHiViT',
        arch='base',
        reconstruction_type='pixel',
        mask_ratio=0.75)
    neck = dict(
        type='iTPNPretrainDecoder',
        num_patches=196,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        decoder_embed_dim=512,
        decoder_depth=6,
        decoder_num_heads=16,
        mlp_ratio=4.,
        reconstruction_type='pixel',
        #  transformer pyramid
        fpn_dim=256,
        fpn_depth=2,
        num_outs=3,
    )
    head = dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        loss=dict(type='PixelReconstructionLoss', criterion='L2'))

    alg = iTPN(
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

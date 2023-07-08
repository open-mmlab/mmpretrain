# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import iTPN
from mmpretrain.structures import DataSample


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

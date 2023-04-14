# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest.mock import MagicMock

import pytest
import torch

from mmpretrain.models import EVA
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_eva():
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
        predict_feature_dim=512,
        mlp_ratio=4.)
    head = dict(
        type='MIMHead',
        loss=dict(
            type='CosineSimilarityLoss', shift_factor=1.0, scale_factor=1.0))

    alg = EVA(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=data_preprocessor)

    target_generator = MagicMock(
        return_value=(torch.ones(2, 197, 512), torch.ones(2, 197, 197)))
    alg.target_generator = target_generator

    fake_data = {
        'inputs': torch.randn((2, 3, 224, 224)),
        'data_samples': [DataSample() for _ in range(2)]
    }
    fake_inputs = alg.data_preprocessor(fake_data)
    fake_outputs = alg(**fake_inputs, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

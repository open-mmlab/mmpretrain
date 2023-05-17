# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
from unittest.mock import MagicMock

import pytest
import torch

from mmpretrain.models import MILAN, MILANViT
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_milan_vit():
    backbone = dict(arch='b', patch_size=16, mask_ratio=0.75)
    milan_backbone = MILANViT(**backbone)
    milan_backbone.init_weights()
    fake_inputs = torch.randn((2, 3, 224, 224))

    # test with mask
    fake_outputs = milan_backbone(fake_inputs,
                                  torch.ones(2, 197, 197)[:, 0, 1:])[0]
    assert list(fake_outputs.shape) == [2, 50, 768]

    # test without mask
    fake_outputs = milan_backbone(fake_inputs, None)
    assert fake_outputs[0].shape == torch.Size([2, 197, 768])


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_milan():
    data_preprocessor = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_rgb': True
    }

    backbone = dict(type='MILANViT', arch='b', patch_size=16, mask_ratio=0.75)
    neck = dict(
        type='MILANPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.)
    head = dict(
        type='MIMHead',
        loss=dict(
            type='CosineSimilarityLoss', shift_factor=2.0, scale_factor=2.0))

    alg = MILAN(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=copy.deepcopy(data_preprocessor))

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

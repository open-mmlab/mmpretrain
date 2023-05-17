# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
from mmengine.utils import digit_version

from mmpretrain.models import MaskFeat, MaskFeatViT
from mmpretrain.structures import DataSample


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_maskfeat_vit():
    maskfeat_backbone = MaskFeatViT()
    maskfeat_backbone.init_weights()
    fake_inputs = torch.randn((2, 3, 224, 224))
    fake_mask = torch.randn((2, 14, 14)).flatten(1).bool()

    # test with mask
    fake_outputs = maskfeat_backbone(fake_inputs, fake_mask)
    assert list(fake_outputs.shape) == [2, 197, 768]

    # test without mask
    fake_outputs = maskfeat_backbone(fake_inputs, None)
    assert fake_outputs[0].shape == torch.Size([2, 197, 768])


@pytest.mark.skipif(
    digit_version(torch.__version__) < digit_version('1.7.0'),
    reason='torch version')
@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_maskfeat():
    data_preprocessor = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_rgb': True
    }

    backbone = dict(type='MaskFeatViT', arch='b', patch_size=16)
    neck = dict(
        type='LinearNeck', in_channels=768, out_channels=108, gap_dim=0)
    head = dict(
        type='MIMHead',
        loss=dict(type='PixelReconstructionLoss', criterion='L2'))
    target_generator = dict(
        type='HOGGenerator', nbins=9, pool=8, gaussian_window=16)

    alg = MaskFeat(
        backbone=backbone,
        neck=neck,
        head=head,
        target_generator=target_generator,
        data_preprocessor=data_preprocessor)

    # test forward_train
    fake_data_sample = DataSample()
    fake_mask = torch.rand((14, 14)).bool()
    fake_data_sample.set_mask(fake_mask)
    fake_data = {
        'inputs': torch.randn((1, 3, 224, 224)),
        'data_samples': [fake_data_sample]
    }

    fake_input = alg.data_preprocessor(fake_data)
    fake_outputs = alg(**fake_input, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

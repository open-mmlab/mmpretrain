# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest import TestCase

import pytest
import torch

from mmpretrain.models import MoCoV3, MoCoV3ViT
from mmpretrain.structures import DataSample


class TestMoCoV3(TestCase):

    backbone = dict(
        type='MoCoV3ViT',
        arch='mocov3-small',  # embed_dim = 384
        patch_size=16,
        frozen_stages=12,
        stop_grad_conv1=True,
        norm_eval=True)
    neck = dict(
        type='NonLinearNeck',
        in_channels=384,
        hid_channels=2,
        out_channels=2,
        num_layers=2,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=False,
        norm_cfg=dict(type='BN1d'))
    head = dict(
        type='MoCoV3Head',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=2,
            hid_channels=2,
            out_channels=2,
            num_layers=2,
            with_bias=False,
            with_last_bn=True,
            with_last_bn_affine=False,
            with_last_bias=False,
            with_avg_pool=False,
            norm_cfg=dict(type='BN1d')),
        loss=dict(type='CrossEntropyLoss', loss_weight=2 * 0.2),
        temperature=0.2)

    @pytest.mark.skipif(
        platform.system() == 'Windows', reason='Windows mem limit')
    def test_vit(self):
        vit = MoCoV3ViT(
            arch='mocov3-small',
            patch_size=16,
            frozen_stages=12,
            stop_grad_conv1=True,
            norm_eval=True)
        vit.init_weights()
        vit.train()

        for p in vit.parameters():
            assert p.requires_grad is False

    @pytest.mark.skipif(
        platform.system() == 'Windows', reason='Windows mem limit')
    def test_mocov3(self):
        data_preprocessor = dict(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            to_rgb=True)
        alg = MoCoV3(
            backbone=self.backbone,
            neck=self.neck,
            head=self.head,
            data_preprocessor=data_preprocessor)

        fake_data = {
            'inputs':
            [torch.randn((2, 3, 224, 224)),
             torch.randn((2, 3, 224, 224))],
            'data_samples': [DataSample() for _ in range(2)]
        }

        fake_inputs = alg.data_preprocessor(fake_data)
        fake_loss = alg(**fake_inputs, mode='loss')
        self.assertGreater(fake_loss['loss'], 0)

        # test extract
        fake_feats = alg(fake_inputs['inputs'][0], mode='tensor')
        self.assertEqual(fake_feats[0].size(), torch.Size([2, 384]))

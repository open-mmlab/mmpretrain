# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest import TestCase

import pytest
import torch

from mmpretrain.models import BEiT, BEiTPretrainViT
from mmpretrain.structures import DataSample


class TestBEiT(TestCase):

    @pytest.mark.skipif(
        platform.system() == 'Windows', reason='Windows mem limit')
    def test_beit_pretrain_vit(self):
        backbone = dict(
            arch='base',
            patch_size=16,
            drop_path_rate=0.1,
            final_norm=True,
            layer_scale_init_value=0.1,
        )

        beit_backbone = BEiTPretrainViT(**backbone)
        beit_backbone.init_weights()

        fake_inputs = torch.randn((2, 3, 224, 224))
        fake_mask = torch.zeros((2, 196))
        fake_mask[:, 75:150] = 1

        # test with mask
        fake_outputs = beit_backbone(fake_inputs, fake_mask)
        assert fake_outputs[0].shape == torch.Size([2, 197, 768])

        # test without mask
        fake_outputs = beit_backbone(fake_inputs, None)
        assert fake_outputs[0].shape == torch.Size([2, 197, 768])

    @pytest.mark.skipif(
        platform.system() == 'Windows', reason='Windows mem limit')
    def test_beitv1(self):
        data_preprocessor = dict(
            type='TwoNormDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            second_mean=[-31.875, -31.875, -31.875],
            second_std=[318.75, 318.75, 318.75],
            to_rgb=True)

        # model settings
        backbone = dict(
            type='BEiTPretrainViT',
            arch='base',
            patch_size=16,
            drop_path_rate=0.1,
            final_norm=True,
            layer_scale_init_value=0.1)
        neck = None
        head = dict(
            type='BEiTV1Head',
            embed_dims=768,
            num_embed=8192,
            loss=dict(type='CrossEntropyLoss'))
        target_generator = dict(type='DALL-E')

        # build model
        model = BEiT(
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

    @pytest.mark.skipif(
        platform.system() == 'Windows', reason='Windows mem limit')
    def test_beitv2(self):
        data_preprocessor = dict(
            type='TwoNormDataPreprocessor',
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            second_mean=(127.5, 127.5, 127.5),
            second_std=(127.5, 127.5, 127.5),
            to_rgb=True)

        # model settings
        vqkd_encoder = dict(
            arch='base',
            img_size=224,
            patch_size=16,
            in_channels=3,
            out_indices=-1,
            drop_rate=0.,
            drop_path_rate=0.,
            norm_cfg=dict(type='LN', eps=1e-6),
            final_norm=True,
            out_type='featmap',
            with_cls_token=True,
            frozen_stages=-1,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            layer_scale_init_value=0.,
            interpolate_mode='bicubic',
            patch_cfg=dict(),
            layer_cfgs=dict(),
            init_cfg=None)

        layer_scale_init_value = 0.1
        drop_path_rate = 0.  # 0. for 300 epochs and 0.1 for 1600 epochs.
        backbone = dict(
            type='BEiTPretrainViT',
            arch='base',
            patch_size=16,
            out_indices=[-4, -1],
            drop_path_rate=drop_path_rate,
            final_norm=False,
            layer_scale_init_value=layer_scale_init_value)
        neck = dict(
            type='BEiTV2Neck',
            num_layers=1,
            early_layers=9,
            backbone_arch='base',
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value)
        head = dict(
            type='BEiTV2Head',
            embed_dims=768,
            num_embed=8192,
            loss=dict(type='CrossEntropyLoss'))
        target_generator = dict(type='VQKD', encoder_config=vqkd_encoder)

        model = BEiT(
            backbone=backbone,
            neck=neck,
            head=head,
            target_generator=target_generator,
            data_preprocessor=data_preprocessor)

        fake_img = torch.rand((1, 3, 224, 224))
        fake_target_img = torch.rand((1, 3, 224, 224))
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
        assert isinstance(fake_outputs['loss_1'].item(), float)
        assert isinstance(fake_outputs['loss_2'].item(), float)

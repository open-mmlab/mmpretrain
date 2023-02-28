# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile
from copy import deepcopy
from unittest import TestCase

import torch
from mmengine.runner import load_checkpoint, save_checkpoint

from mmcls.models.backbones import BEiTAdapter, VitAdapter
from .utils import timm_resize_pos_embed


class TestVitAdapter(TestCase):
    CUSTOM_ARCH = {
        'embed_dims': 32,
        'num_layers': 10,
        'num_heads': 16,
        'feedforward_channels': 512,
        'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 15]],
        'window_size': [14, 14, 0, 14, 0, 14, 0, 14, 14, 0],
        'value_proj_ratio': 1.0
    }

    def setUp(self):
        self.cfg = dict(
            arch='deit-t', img_size=224, patch_size=16, drop_path_rate=0.1)

    def test_structure(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            VitAdapter(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch setting'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = deepcopy(self.CUSTOM_ARCH)
            cfg['arch'].pop('window_size')
            VitAdapter(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = deepcopy(self.CUSTOM_ARCH)
        cfg['deform_num_heads'] = 16
        model = VitAdapter(**cfg)
        self.assertEqual(model.embed_dims, 32)
        self.assertEqual(model.num_layers, 10)
        for layer in model.layers:
            self.assertEqual(layer.attn.num_heads, 16)
            self.assertEqual(layer.ffn.feedforward_channels, 512)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = VitAdapter(**cfg)
        self.assertEqual(len(model.layers), 12)
        dpr_inc = 0.1 / (12 - 1)
        dpr = 0
        for layer in model.layers:
            self.assertEqual(layer.attn.embed_dims, 192)
            self.assertEqual(layer.attn.num_heads, 3)
            self.assertEqual(layer.ffn.feedforward_channels, 768)
            self.assertAlmostEqual(layer.attn.out_drop.drop_prob, dpr)
            self.assertAlmostEqual(layer.ffn.dropout_layer.drop_prob, dpr)
            dpr += dpr_inc

    def test_init_weights(self):
        # test weight init cfg
        cfg = deepcopy(self.cfg)
        cfg['init_cfg'] = [
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]
        model = VitAdapter(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()
        # The pos_embed is all zero before initialize
        self.assertTrue(torch.allclose(model.pos_embed, torch.tensor(0.)))

        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))
        self.assertFalse(torch.allclose(model.pos_embed, torch.tensor(0.)))

        # test load checkpoint
        pretrain_pos_embed = model.pos_embed.clone().detach()
        tmpdir = tempfile.gettempdir()
        checkpoint = os.path.join(tmpdir, 'test.pth')
        save_checkpoint(model.state_dict(), checkpoint)
        cfg = deepcopy(self.cfg)
        model = VitAdapter(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        self.assertTrue(torch.allclose(model.pos_embed, pretrain_pos_embed))

        # test load checkpoint with different img_size
        cfg = deepcopy(self.cfg)
        cfg['img_size'] = 384
        model = VitAdapter(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        resized_pos_embed = timm_resize_pos_embed(pretrain_pos_embed,
                                                  model.pos_embed)
        self.assertTrue(torch.allclose(model.pos_embed, resized_pos_embed))

        os.remove(checkpoint)

    def test_forward(self):
        imgs = torch.randn(1, 3, 64, 64)

        # Test forward
        cfg = deepcopy(self.cfg)
        model = VitAdapter(**cfg)
        model.eval()
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for stride, out in zip([1, 2, 4, 8], outs):
            self.assertEqual(out.shape, (1, 192, 16 // stride, 16 // stride))

        # Test forward with dynamic input size
        imgs = torch.randn(1, 3, 256, 309)
        cfg = deepcopy(self.cfg)
        model = VitAdapter(**cfg)
        model.eval()
        for imgs in [imgs]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 4)
            expect_feat_shape = (math.ceil(imgs.shape[2] / 32),
                                 math.ceil(imgs.shape[3] / 32))
            self.assertEqual(outs[-1].shape, (1, 192, *expect_feat_shape))


class TestBEiTAdapter(TestCase):
    CUSTOM_ARCH = {
        'embed_dims': 32,
        'num_layers': 10,
        'num_heads': 8,
        'feedforward_channels': 512,
        'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 15]],
        'window_size': [14, 14, 56, 14, 56, 14, 56, 14, 14, 56],
        'value_proj_ratio': 1.0
    }

    def setUp(self):
        self.cfg = dict(
            arch='b', img_size=224, patch_size=16, drop_path_rate=0.1)

    def test_structure(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            BEiTAdapter(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch settings'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = deepcopy(self.CUSTOM_ARCH)
            cfg['arch'].pop('window_size')
            BEiTAdapter(**cfg)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = BEiTAdapter(**cfg)
        self.assertEqual(len(model.layers), 12)
        dpr_inc = 0.1 / (12 - 1)
        dpr = 0
        for layer in model.layers:
            self.assertEqual(layer.attn.embed_dims, 768)
            self.assertEqual(layer.attn.num_heads, 12)
            self.assertEqual(layer.ffn.feedforward_channels, 3072)
            self.assertAlmostEqual(layer.drop_path.drop_prob, dpr)
            self.assertAlmostEqual(layer.ffn.dropout_layer.drop_prob, dpr)
            dpr += dpr_inc

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = self.CUSTOM_ARCH
        cfg['deform_num_heads'] = 16
        model = BEiTAdapter(**cfg)
        self.assertEqual(model.embed_dims, 32)
        self.assertEqual(model.num_layers, 10)
        for layer in model.layers:
            self.assertEqual(layer.attn.num_heads, 8)
            self.assertEqual(layer.ffn.feedforward_channels, 512)

    def test_init_weights(self):
        # test weight init cfg
        cfg = deepcopy(self.cfg)
        cfg['init_cfg'] = [
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]
        cfg['arch'] = self.CUSTOM_ARCH
        cfg['deform_num_heads'] = 16
        model = BEiTAdapter(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()

        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))

        # test load checkpoint
        pretrain_patch_embed = model.patch_embed.projection.weight.clone(
        ).detach()
        tmpdir = tempfile.gettempdir()
        checkpoint = os.path.join(tmpdir, 'test.pth')
        save_checkpoint(model.state_dict(), checkpoint)
        cfg = deepcopy(self.cfg)
        cfg['arch'] = self.CUSTOM_ARCH
        cfg['deform_num_heads'] = 16
        model = BEiTAdapter(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        self.assertTrue(
            torch.allclose(model.patch_embed.projection.weight,
                           pretrain_patch_embed))

        os.remove(checkpoint)

    def test_forward(self):
        imgs = torch.randn(1, 3, 64, 64)

        # Test forward
        cfg = deepcopy(self.cfg)
        cfg['arch'] = self.CUSTOM_ARCH
        cfg['deform_num_heads'] = 16
        model = BEiTAdapter(**cfg)
        model.eval()
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for stride, out in zip([1, 2, 4, 8], outs):
            self.assertEqual(out.shape, (1, 32, 16 // stride, 16 // stride))

        # Test forward with layer scale
        cfg = deepcopy(self.cfg)
        cfg['arch'] = self.CUSTOM_ARCH
        cfg['deform_num_heads'] = 16
        cfg['layer_scale_init_value'] = 1.0
        model = BEiTAdapter(**cfg)
        model.eval()
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for stride, out in zip([1, 2, 4, 8], outs):
            self.assertEqual(out.shape, (1, 32, 16 // stride, 16 // stride))

        # Test forward with dynamic input size
        imgs = torch.randn(1, 3, 256, 309)
        cfg['arch'] = self.CUSTOM_ARCH
        cfg['deform_num_heads'] = 16
        model = BEiTAdapter(**cfg)
        model.eval()
        for imgs in [imgs]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 4)
            expect_feat_shape = (math.ceil(imgs.shape[2] / 32),
                                 math.ceil(imgs.shape[3] / 32))
            self.assertEqual(outs[-1].shape, (1, 32, *expect_feat_shape))

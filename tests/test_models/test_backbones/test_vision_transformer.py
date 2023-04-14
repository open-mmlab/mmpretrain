# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile
from copy import deepcopy
from unittest import TestCase

import torch
from mmengine.runner import load_checkpoint, save_checkpoint

from mmpretrain.models.backbones import VisionTransformer
from .utils import timm_resize_pos_embed


class TestVisionTransformer(TestCase):

    def setUp(self):
        self.cfg = dict(
            arch='b', img_size=224, patch_size=16, drop_path_rate=0.1)

    def test_structure(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            VisionTransformer(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }
            VisionTransformer(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 128,
            'num_layers': 24,
            'num_heads': 16,
            'feedforward_channels': 1024
        }
        model = VisionTransformer(**cfg)
        self.assertEqual(model.embed_dims, 128)
        self.assertEqual(model.num_layers, 24)
        for layer in model.layers:
            self.assertEqual(layer.attn.num_heads, 16)
            self.assertEqual(layer.ffn.feedforward_channels, 1024)

        # Test out_indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = {1: 1}
        with self.assertRaisesRegex(AssertionError, "get <class 'dict'>"):
            VisionTransformer(**cfg)
        cfg['out_indices'] = [0, 13]
        with self.assertRaisesRegex(AssertionError, 'Invalid out_indices 13'):
            VisionTransformer(**cfg)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = VisionTransformer(**cfg)
        self.assertEqual(len(model.layers), 12)
        dpr_inc = 0.1 / (12 - 1)
        dpr = 0
        for layer in model.layers:
            self.assertEqual(layer.attn.embed_dims, 768)
            self.assertEqual(layer.attn.num_heads, 12)
            self.assertEqual(layer.ffn.feedforward_channels, 3072)
            self.assertAlmostEqual(layer.attn.out_drop.drop_prob, dpr)
            self.assertAlmostEqual(layer.ffn.dropout_layer.drop_prob, dpr)
            dpr += dpr_inc

        # Test model structure:  prenorm
        cfg = deepcopy(self.cfg)
        cfg['pre_norm'] = True
        model = VisionTransformer(**cfg)
        self.assertNotEqual(model.pre_norm.__class__, torch.nn.Identity)

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
        model = VisionTransformer(**cfg)
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
        model = VisionTransformer(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        self.assertTrue(torch.allclose(model.pos_embed, pretrain_pos_embed))

        # test load checkpoint with different img_size
        cfg = deepcopy(self.cfg)
        cfg['img_size'] = 384
        model = VisionTransformer(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        resized_pos_embed = timm_resize_pos_embed(pretrain_pos_embed,
                                                  model.pos_embed)
        self.assertTrue(torch.allclose(model.pos_embed, resized_pos_embed))

        os.remove(checkpoint)

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)

        # test with_cls_token=False
        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['out_type'] = 'cls_token'
        with self.assertRaisesRegex(ValueError, 'must be True'):
            VisionTransformer(**cfg)

        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['out_type'] = 'featmap'
        model = VisionTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 768, 14, 14))

        # test with output cls_token
        cfg = deepcopy(self.cfg)
        model = VisionTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        cls_token = outs[-1]
        self.assertEqual(cls_token.shape, (1, 768))

        # Test forward with multi out indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = [-3, -2, -1]
        model = VisionTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 3)
        for out in outs:
            self.assertEqual(out.shape, (1, 768))

        # Test forward with dynamic input size
        imgs1 = torch.randn(1, 3, 224, 224)
        imgs2 = torch.randn(1, 3, 256, 256)
        imgs3 = torch.randn(1, 3, 256, 309)
        cfg = deepcopy(self.cfg)
        cfg['out_type'] = 'featmap'
        model = VisionTransformer(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            patch_token = outs[-1]
            expect_feat_shape = (math.ceil(imgs.shape[2] / 16),
                                 math.ceil(imgs.shape[3] / 16))
            self.assertEqual(patch_token.shape, (1, 768, *expect_feat_shape))

# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile
from copy import deepcopy
from unittest import TestCase

import torch
from mmengine.runner import load_checkpoint, save_checkpoint
from torch import nn

from mmcls.models.backbones import GPViT
from mmcls.models.backbones.gpvit import GPBlock, LePEAttentionDWBlock
from mmcls.models.utils import resize_pos_embed


class TestGPViT(TestCase):

    def setUp(self):
        self.cfg = dict(
            arch='L1', img_size=224, patch_size=8, drop_path_rate=0.1)

    def test_structure(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            GPViT(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'embed_dims': 128,
                'window_size': 2,
                'num_heads': 16,
            }
            GPViT(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 128,
            'window_size': 2,
            'num_layers': 12,
            'num_heads': 12,
            'num_group_heads': 6,
            'num_ungroup_heads': 6,
            'ffn_ratio': 4.,
            'num_convs': 0,
            'mlpmixer_depth': 1,
            'group_layers': {
                1: 64,
                4: 32,
                7: 32,
                10: 16
            },
        }
        model = GPViT(**cfg)
        self.assertEqual(model.embed_dims, 128)
        self.assertEqual(model.num_layers, 12)

        # Test out_indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = {1: 1}
        with self.assertRaisesRegex(AssertionError, "get <class 'dict'>"):
            GPViT(**cfg)
        cfg['out_indices'] = [0, 13]
        with self.assertRaisesRegex(AssertionError, 'Invalid out_indices 13'):
            GPViT(**cfg)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = GPViT(**cfg)
        self.assertEqual(len(model.layers), 12)
        dpr_inc = 0.1 / (12 - 1)
        dpr = 0
        for layer in model.layers:
            if isinstance(layer, LePEAttentionDWBlock):
                self.assertEqual(layer.embed_dims, 216)
                self.assertEqual(layer.num_heads, 12)
                self.assertEqual(layer.ffn.feedforward_channels, 216 * 4)
                if dpr == 0:
                    self.assertIsInstance(layer.drop_path, nn.Identity)
                else:
                    self.assertAlmostEqual(layer.drop_path.drop_prob, dpr)
            elif isinstance(layer, GPBlock):
                self.assertEqual(layer.group_layer.attn.embed_dims, 216)
                self.assertEqual(layer.group_layer.attn.num_heads, 6)
                self.assertAlmostEqual(
                    layer.un_group_layer.drop_path.drop_prob, dpr)
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
        model = GPViT(**cfg)
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
        model = GPViT(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        self.assertTrue(torch.allclose(model.pos_embed, pretrain_pos_embed))

        # test load checkpoint with different img_size
        cfg = deepcopy(self.cfg)
        cfg['img_size'] = 384
        model = GPViT(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        ckpt_pos_embed_shape = (224 // 8, 224 // 8)
        pos_embed_shape = model.patch_embed.init_out_size
        resized_pos_embed = resize_pos_embed(pretrain_pos_embed,
                                             ckpt_pos_embed_shape,
                                             pos_embed_shape, 'bicubic', 0)
        self.assertTrue(torch.allclose(model.pos_embed, resized_pos_embed))

        os.remove(checkpoint)

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)

        # test forward
        cfg = deepcopy(self.cfg)
        model = GPViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 216, 28, 28))

        # Test forward with multi out indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = [-3, -2, -1]
        model = GPViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 3)
        for out in outs:
            self.assertEqual(out.shape, (1, 216, 28, 28))

        # Test forward with dynamic input size
        imgs1 = torch.randn(1, 3, 224, 224)
        imgs2 = torch.randn(1, 3, 256, 256)
        imgs3 = torch.randn(1, 3, 256, 224)
        cfg = deepcopy(self.cfg)
        model = GPViT(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            expect_feat_shape = (math.ceil(imgs.shape[2] / 8),
                                 math.ceil(imgs.shape[3] / 8))
            self.assertEqual(outs[-1].shape, (1, 216, *expect_feat_shape))

        with self.assertRaisesRegex(AssertionError, 'need to be dibided by'):
            imgs = torch.randn(1, 3, 1333, 309)
            outs = model(imgs)

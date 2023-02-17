# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from copy import deepcopy
from unittest import TestCase

import torch
from mmengine.runner import load_checkpoint, save_checkpoint

from mmpretrain.models.backbones import RevVisionTransformer
from .utils import timm_resize_pos_embed


class TestRevVisionTransformer(TestCase):

    def setUp(self):
        self.cfg = dict(
            arch='b', img_size=224, patch_size=16, drop_path_rate=0.1)

    def test_structure(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            RevVisionTransformer(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }
            RevVisionTransformer(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 128,
            'num_layers': 24,
            'num_heads': 16,
            'feedforward_channels': 1024
        }
        model = RevVisionTransformer(**cfg)
        self.assertEqual(model.embed_dims, 128)
        self.assertEqual(model.num_layers, 24)
        for layer in model.layers:
            self.assertEqual(layer.attn.num_heads, 16)
            self.assertEqual(layer.ffn.feedforward_channels, 1024)

        # Test out_indices
        # TODO: to be implemented, current only support last layer
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = {1: 1}
        with self.assertRaisesRegex(AssertionError, "get <class 'dict'>"):
            RevVisionTransformer(**cfg)
        cfg['out_indices'] = [13]
        with self.assertRaisesRegex(AssertionError, 'Invalid out_indices 13'):
            RevVisionTransformer(**cfg)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = RevVisionTransformer(**cfg)
        self.assertEqual(len(model.layers), 12)
        dpr_inc = 0.1 / (12 - 1)
        dpr = 0
        for layer in model.layers:
            self.assertEqual(layer.attn.embed_dims, 768)
            self.assertEqual(layer.attn.num_heads, 12)
            self.assertEqual(layer.ffn.feedforward_channels, 3072)
            # self.assertAlmostEqual(layer.attn.out_drop.drop_prob, dpr)
            # self.assertAlmostEqual(layer.ffn.dropout_layer.drop_prob, dpr)
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
        model = RevVisionTransformer(**cfg)
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
        model = RevVisionTransformer(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        self.assertTrue(torch.allclose(model.pos_embed, pretrain_pos_embed))

        # test load checkpoint with different img_size
        cfg = deepcopy(self.cfg)
        cfg['img_size'] = 384
        model = RevVisionTransformer(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        resized_pos_embed = timm_resize_pos_embed(pretrain_pos_embed,
                                                  model.pos_embed)
        self.assertTrue(torch.allclose(model.pos_embed, resized_pos_embed))

        os.remove(checkpoint)

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)

        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['output_cls_token'] = False
        model = RevVisionTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 768 * 2))

        # Test forward with dynamic input size
        imgs1 = torch.randn(1, 3, 224, 224)
        imgs2 = torch.randn(1, 3, 256, 256)
        imgs3 = torch.randn(1, 3, 256, 309)
        cfg = deepcopy(self.cfg)
        model = RevVisionTransformer(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            avg_token = outs[-1]
            self.assertEqual(avg_token.shape, (1, 768 * 2))

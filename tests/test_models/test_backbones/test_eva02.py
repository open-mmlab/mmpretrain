# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch

from mmpretrain.models.backbones import ViTEVA02


class TestEVA02(TestCase):

    def setUp(self):
        self.cfg = dict(
            arch='t',
            img_size=336,
            patch_size=14,
            drop_path_rate=0.1,
            drop_rate=0.1,
            attn_drop_rate=0.2,
            proj_drop_rate=0.3,
        )

    def test_structure(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            ViTEVA02(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': int(24 * 4 * 2 / 3)
            }
            ViTEVA02(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 128,
            'num_layers': 6,
            'num_heads': 16,
            'feedforward_channels': int(128 * 4 * 2 / 3)
        }
        model = ViTEVA02(**cfg)
        self.assertEqual(model.embed_dims, 128)
        self.assertEqual(model.num_layers, 6)
        for layer in model.layers:
            self.assertEqual(layer.attn.num_heads, 16)

        # Test out_indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = {1: 1}
        with self.assertRaisesRegex(AssertionError, "get <class 'dict'>"):
            ViTEVA02(**cfg)
        cfg['out_indices'] = [0, 13]
        with self.assertRaisesRegex(AssertionError, 'Invalid out_indices 13'):
            ViTEVA02(**cfg)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = ViTEVA02(**cfg)
        self.assertEqual(len(model.layers), 12)
        self.assertEqual(model.cls_token.shape, (1, 1, 192))
        self.assertEqual(model.pos_embed.shape, (1, 577, 192))
        dpr_inc = 0.1 / (12 - 1)
        dpr = 0
        for layer in model.layers:
            self.assertEqual(layer.attn.embed_dims, 192)
            self.assertEqual(layer.attn.num_heads, 3)
            self.assertAlmostEqual(layer.drop_path.drop_prob, dpr)
            self.assertAlmostEqual(layer.mlp.dropout_layer.p, 0.1)
            self.assertAlmostEqual(layer.attn.attn_drop.p, 0.2)
            self.assertAlmostEqual(layer.attn.proj_drop.p, 0.3)
            dpr += dpr_inc

        # Test model structure: final_norm
        cfg = deepcopy(self.cfg)
        cfg['final_norm'] = True
        model = ViTEVA02(**cfg)
        self.assertNotEqual(model.norm1.__class__, torch.nn.Identity)

    def test_forward(self):
        imgs = torch.randn(1, 3, 336, 336)

        # test with_cls_token=False
        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['out_type'] = 'cls_token'
        with self.assertRaisesRegex(ValueError, 'must be True'):
            ViTEVA02(**cfg)

        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['out_type'] = 'raw'
        model = ViTEVA02(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 24 * 24, 192))

        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['out_type'] = 'featmap'
        model = ViTEVA02(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 192, 24, 24))

        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['out_type'] = 'avg_featmap'
        model = ViTEVA02(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 192))

        # test with output cls_token
        cfg = deepcopy(self.cfg)
        model = ViTEVA02(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        cls_token = outs[-1]
        self.assertEqual(cls_token.shape, (1, 192))

        # Test forward with multi out indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = [-3, -2, -1]
        model = ViTEVA02(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 3)
        for out in outs:
            self.assertEqual(out.shape, (1, 192))

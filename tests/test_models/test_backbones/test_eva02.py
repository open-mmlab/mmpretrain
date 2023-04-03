# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from unittest import TestCase

import torch

from mmpretrain.models.backbones import EVA02


class TestEVA02(TestCase):

    def setUp(self):
        self.cfg = dict(
            arch='b',
            img_size=448,
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
            EVA02(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'num_layers': 24,
                'num_heads': 16,
                'mlp_ratio': 4 * 2 / 3
            }
            EVA02(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 128,
            'num_layers': 24,
            'num_heads': 16,
            'mlp_ratio': 4
        }
        model = EVA02(**cfg)
        self.assertEqual(model.embed_dims, 128)
        self.assertEqual(model.num_layers, 24)
        for layer in model.layers:
            self.assertEqual(layer.attn.num_heads, 16)
            self.assertEqual(layer.mlp.hidden_features, 128 * 4)

        # Test out_indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = {1: 1}
        with self.assertRaisesRegex(AssertionError, "get <class 'dict'>"):
            EVA02(**cfg)
        cfg['out_indices'] = [0, 13]
        with self.assertRaisesRegex(AssertionError, 'Invalid out_indices 13'):
            EVA02(**cfg)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = EVA02(**cfg)
        self.assertEqual(len(model.layers), 12)
        self.assertEqual(model.cls_token.shape, (1, 1, 768))
        self.assertEqual(model.pos_embed.shape, (1, 1025, 768))
        self.assertEqual(model.rope.freqs_cos.shape, (1024, 64))
        self.assertEqual(model.rope.freqs_sin.shape, (1024, 64))
        dpr_inc = 0.1 / (12 - 1)
        dpr = 0
        for layer in model.layers:
            self.assertEqual(layer.attn.embed_dims, 768)
            self.assertEqual(layer.attn.num_heads, 12)
            self.assertEqual(layer.mlp.hidden_features, 2048)
            self.assertAlmostEqual(layer.drop_path.drop_prob, dpr)
            self.assertAlmostEqual(layer.mlp.dropout_layer.p, 0.1)
            self.assertAlmostEqual(layer.attn.attn_drop.p, 0.2)
            self.assertAlmostEqual(layer.attn.proj_drop.p, 0.3)
            self.assertEqual(layer.attn.rope.freqs_cos.shape, (1024, 64))
            self.assertEqual(layer.attn.rope.freqs_sin.shape, (1024, 64))
            dpr += dpr_inc

        # Test model structure: final_norm
        cfg = deepcopy(self.cfg)
        cfg['final_norm'] = True
        model = EVA02(**cfg)
        self.assertNotEqual(model.final_norm.__class__, torch.nn.Identity)

        cfg = deepcopy(self.cfg)
        cfg['final_norm'] = False
        model = EVA02(**cfg)
        self.assertEqual(model.final_norm.__class__, torch.nn.Identity)

    def test_forward(self):
        imgs = torch.randn(1, 3, 448, 448)

        # test with_cls_token=False
        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['out_type'] = 'cls_token'
        with self.assertRaisesRegex(ValueError, 'must be True'):
            EVA02(**cfg)

        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['out_type'] = 'raw'
        model = EVA02(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 32 * 32, 768))

        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['out_type'] = 'featmap'
        model = EVA02(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 768, 32, 32))

        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['out_type'] = 'avg_featmap'
        model = EVA02(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 768))

        # test with output cls_token
        cfg = deepcopy(self.cfg)
        model = EVA02(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        cls_token = outs[-1]
        self.assertEqual(cls_token.shape, (1, 768))

        # Test forward with multi out indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = [-3, -2, -1]
        model = EVA02(**cfg)
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
        model = EVA02(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            patch_token = outs[-1]
            expect_feat_shape = (math.ceil(imgs.shape[2] / 14),
                                 math.ceil(imgs.shape[3] / 14))
            self.assertEqual(patch_token.shape, (1, 768, *expect_feat_shape))

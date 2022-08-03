# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch

from mmcls.models.backbones import MViT


class TestMViT(TestCase):

    def setUp(self):
        self.cfg = dict(arch='tiny', img_size=224, drop_path_rate=0.1)

    def test_arch(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            MViT(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'embed_dims': 96,
                'num_layers': 10,
            }
            MViT(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        embed_dims = 96
        num_layers = 10
        num_heads = 1
        downscale_indices = (2, 5, 7)
        cfg['arch'] = {
            'embed_dims': embed_dims,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'downscale_indices': downscale_indices
        }
        model = MViT(**cfg)
        self.assertEqual(len(model.blocks), num_layers)
        for i, block in enumerate(model.blocks):
            if i in downscale_indices:
                num_heads *= 2
                embed_dims *= 2
            self.assertEqual(block.out_dims, embed_dims)
            self.assertEqual(block.attn.num_heads, num_heads)

    def test_init_weights(self):
        # test weight init cfg
        cfg = deepcopy(self.cfg)
        cfg['use_abs_pos_embed'] = True
        cfg['init_cfg'] = [
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]
        model = MViT(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()
        # The pos_embed is all zero before initialize
        self.assertTrue(torch.allclose(model.pos_embed, torch.tensor(0.)))

        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))
        self.assertFalse(torch.allclose(model.pos_embed, torch.tensor(0.)))
        self.assertFalse(
            torch.allclose(model.blocks[0].attn.rel_pos_h, torch.tensor(0.)))
        self.assertFalse(
            torch.allclose(model.blocks[0].attn.rel_pos_w, torch.tensor(0.)))

        # test rel_pos_zero_init
        cfg = deepcopy(self.cfg)
        cfg['rel_pos_zero_init'] = True
        model = MViT(**cfg)
        model.init_weights()
        self.assertTrue(
            torch.allclose(model.blocks[0].attn.rel_pos_h, torch.tensor(0.)))
        self.assertTrue(
            torch.allclose(model.blocks[0].attn.rel_pos_w, torch.tensor(0.)))

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)

        cfg = deepcopy(self.cfg)
        model = MViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (1, 768, 7, 7))

        # test multiple output indices
        cfg = deepcopy(self.cfg)
        cfg['out_scales'] = (0, 1, 2, 3)
        model = MViT(**cfg)
        model.init_weights()
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for stride, out in zip([1, 2, 4, 8], outs):
            self.assertEqual(out.shape,
                             (1, 96 * stride, 56 // stride, 56 // stride))

        # test dim_mul_in_attention = False
        cfg = deepcopy(self.cfg)
        cfg['out_scales'] = (0, 1, 2, 3)
        cfg['dim_mul_in_attention'] = False
        model = MViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for dim_mul, stride, out in zip([2, 4, 8, 8], [1, 2, 4, 8], outs):
            self.assertEqual(out.shape,
                             (1, 96 * dim_mul, 56 // stride, 56 // stride))

        # test rel_pos_spatial = False
        cfg = deepcopy(self.cfg)
        cfg['out_scales'] = (0, 1, 2, 3)
        cfg['rel_pos_spatial'] = False
        cfg['img_size'] = None
        model = MViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for stride, out in zip([1, 2, 4, 8], outs):
            self.assertEqual(out.shape,
                             (1, 96 * stride, 56 // stride, 56 // stride))

        # test residual_pooling = False
        cfg = deepcopy(self.cfg)
        cfg['out_scales'] = (0, 1, 2, 3)
        cfg['residual_pooling'] = False
        model = MViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for stride, out in zip([1, 2, 4, 8], outs):
            self.assertEqual(out.shape,
                             (1, 96 * stride, 56 // stride, 56 // stride))

        # test use_abs_pos_embed = True
        cfg = deepcopy(self.cfg)
        cfg['out_scales'] = (0, 1, 2, 3)
        cfg['use_abs_pos_embed'] = True
        model = MViT(**cfg)
        model.init_weights()
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for stride, out in zip([1, 2, 4, 8], outs):
            self.assertEqual(out.shape,
                             (1, 96 * stride, 56 // stride, 56 // stride))

        # test dynamic inputs shape
        cfg = deepcopy(self.cfg)
        cfg['out_scales'] = (0, 1, 2, 3)
        model = MViT(**cfg)
        imgs = torch.randn(1, 3, 352, 260)
        h_resolution = (352 + 2 * 3 - 7) // 4 + 1
        w_resolution = (260 + 2 * 3 - 7) // 4 + 1
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        expect_h = h_resolution
        expect_w = w_resolution
        for i, out in enumerate(outs):
            self.assertEqual(out.shape, (1, 96 * 2**i, expect_h, expect_w))
            expect_h = (expect_h + 2 * 1 - 3) // 2 + 1
            expect_w = (expect_w + 2 * 1 - 3) // 2 + 1

    def test_structure(self):
        # test drop_path_rate decay
        cfg = deepcopy(self.cfg)
        cfg['drop_path_rate'] = 0.2
        model = MViT(**cfg)
        for i, block in enumerate(model.blocks):
            expect_prob = 0.2 / (model.num_layers - 1) * i
            if expect_prob > 0:
                self.assertAlmostEqual(block.drop_path.drop_prob, expect_prob)

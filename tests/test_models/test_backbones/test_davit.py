# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch

from mmpretrain.models.backbones import DaViT
from mmpretrain.models.backbones.davit import SpatialBlock


class TestDaViT(TestCase):

    def setUp(self):
        self.cfg = dict(arch='t', patch_size=4, drop_path_rate=0.1)

    def test_structure(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            DaViT(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }
            DaViT(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 64,
            'num_heads': [3, 3, 3, 3],
            'depths': [1, 1, 2, 1]
        }
        model = DaViT(**cfg)
        self.assertEqual(model.embed_dims, 64)
        self.assertEqual(model.num_layers, 4)
        for layer in model.stages:
            self.assertEqual(
                layer.blocks[0].spatial_block.attn.w_msa.num_heads, 3)

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
        model = DaViT(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()

        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)

        cfg = deepcopy(self.cfg)
        model = DaViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0].shape, (1, 768, 7, 7))

        # Test forward with multi out indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = [2, 3]
        model = DaViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 2)
        self.assertEqual(outs[0].shape, (1, 384, 14, 14))
        self.assertEqual(outs[1].shape, (1, 768, 7, 7))

        # test with checkpoint forward
        cfg = deepcopy(self.cfg)
        cfg['with_cp'] = True
        model = DaViT(**cfg)
        for m in model.modules():
            if isinstance(m, SpatialBlock):
                self.assertTrue(m.with_cp)
        model.init_weights()
        model.train()

        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0].shape, (1, 768, 7, 7))

        # Test forward with dynamic input size
        imgs1 = torch.randn(1, 3, 224, 224)
        imgs2 = torch.randn(1, 3, 256, 256)
        imgs3 = torch.randn(1, 3, 256, 309)
        cfg = deepcopy(self.cfg)
        model = DaViT(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            expect_feat_shape = (imgs.shape[2] // 32, imgs.shape[3] // 32)
            self.assertEqual(outs[0].shape, (1, 768, *expect_feat_shape))

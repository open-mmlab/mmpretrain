# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from unittest import TestCase

import torch

from mmcls.models.backbones import DaViT


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
        self.assertEqual(model.num_layers, 5)
        for layer in model.layers:
            self.assertEqual(layer.attn.num_heads, 3)

        # Test out_indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = {1: 1}
        with self.assertRaisesRegex(AssertionError, "get <class 'dict'>"):
            DaViT(**cfg)
        cfg['out_indices'] = [0, 13]
        with self.assertRaisesRegex(AssertionError, 'Invalid out_indices 13'):
            DaViT(**cfg)

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
        self.assertEqual(outs.shape, (1, 512, 14, 14))

        # Test forward with multi out indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = [2, 3]
        model = DaViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 2)
        self.assertEqual(outs[0].shape, (1, 256, 28, 28))
        self.assertEqual(outs[1].shape, (1, 512, 14, 14))

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
            expect_feat_shape = (math.ceil(imgs.shape[2] / 16),
                                 math.ceil(imgs.shape[3] / 16))
            self.assertEqual(outs[0].shape, (1, 512, *expect_feat_shape))

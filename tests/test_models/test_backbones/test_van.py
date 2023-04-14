# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from itertools import chain
from unittest import TestCase

import torch
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch import nn

from mmpretrain.models.backbones import VAN


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


class TestVAN(TestCase):

    def setUp(self):
        self.cfg = dict(arch='t', drop_path_rate=0.1)

    def test_arch(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            VAN(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'embed_dims': [32, 64, 160, 256],
                'ffn_ratios': [8, 8, 4, 4],
            }
            VAN(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        embed_dims = [32, 64, 160, 256]
        depths = [3, 3, 5, 2]
        ffn_ratios = [8, 8, 4, 4]
        cfg['arch'] = {
            'embed_dims': embed_dims,
            'depths': depths,
            'ffn_ratios': ffn_ratios
        }
        model = VAN(**cfg)

        for i in range(len(depths)):
            stage = getattr(model, f'blocks{i + 1}')
            self.assertEqual(stage[-1].out_channels, embed_dims[i])
            self.assertEqual(len(stage), depths[i])

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
        model = VAN(**cfg)
        ori_weight = model.patch_embed1.projection.weight.clone().detach()

        model.init_weights()
        initialized_weight = model.patch_embed1.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))

    def test_forward(self):
        imgs = torch.randn(3, 3, 224, 224)

        cfg = deepcopy(self.cfg)
        model = VAN(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (3, 256, 7, 7))

        # test with patch_sizes
        cfg = deepcopy(self.cfg)
        cfg['patch_sizes'] = [7, 5, 5, 5]
        model = VAN(**cfg)
        outs = model(torch.randn(3, 3, 224, 224))
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (3, 256, 3, 3))

        # test multiple output indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = (0, 1, 2, 3)
        model = VAN(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for emb_size, stride, out in zip([32, 64, 160, 256], [1, 2, 4, 8],
                                         outs):
            self.assertEqual(out.shape,
                             (3, emb_size, 56 // stride, 56 // stride))

        # test with dynamic input shape
        imgs1 = torch.randn(3, 3, 224, 224)
        imgs2 = torch.randn(3, 3, 256, 256)
        imgs3 = torch.randn(3, 3, 256, 309)
        cfg = deepcopy(self.cfg)
        model = VAN(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            feat = outs[-1]
            expect_feat_shape = (math.ceil(imgs.shape[2] / 32),
                                 math.ceil(imgs.shape[3] / 32))
            self.assertEqual(feat.shape, (3, 256, *expect_feat_shape))

    def test_structure(self):
        # test drop_path_rate decay
        cfg = deepcopy(self.cfg)
        cfg['drop_path_rate'] = 0.2
        model = VAN(**cfg)
        depths = model.arch_settings['depths']
        stages = [model.blocks1, model.blocks2, model.blocks3, model.blocks4]
        blocks = chain(*[stage for stage in stages])
        total_depth = sum(depths)
        dpr = [
            x.item()
            for x in torch.linspace(0, cfg['drop_path_rate'], total_depth)
        ]
        for i, (block, expect_prob) in enumerate(zip(blocks, dpr)):
            if expect_prob == 0:
                assert isinstance(block.drop_path, nn.Identity)
            else:
                self.assertAlmostEqual(block.drop_path.drop_prob, expect_prob)

        # test VAN with norm_eval=True
        cfg = deepcopy(self.cfg)
        cfg['norm_eval'] = True
        cfg['norm_cfg'] = dict(type='BN')
        model = VAN(**cfg)
        model.init_weights()
        model.train()
        self.assertTrue(check_norm_state(model.modules(), False))

        # test VAN with first stage frozen.
        cfg = deepcopy(self.cfg)
        frozen_stages = 0
        cfg['frozen_stages'] = frozen_stages
        cfg['out_indices'] = (0, 1, 2, 3)
        model = VAN(**cfg)
        model.init_weights()
        model.train()

        # the patch_embed and first stage should not require grad.
        self.assertFalse(model.patch_embed1.training)
        for param in model.patch_embed1.parameters():
            self.assertFalse(param.requires_grad)
        for i in range(frozen_stages + 1):
            patch = getattr(model, f'patch_embed{i+1}')
            for param in patch.parameters():
                self.assertFalse(param.requires_grad)
            blocks = getattr(model, f'blocks{i + 1}')
            for param in blocks.parameters():
                self.assertFalse(param.requires_grad)
            norm = getattr(model, f'norm{i + 1}')
            for param in norm.parameters():
                self.assertFalse(param.requires_grad)

        # the second stage should require grad.
        for i in range(frozen_stages + 1, 4):
            patch = getattr(model, f'patch_embed{i + 1}')
            for param in patch.parameters():
                self.assertTrue(param.requires_grad)
            blocks = getattr(model, f'blocks{i+1}')
            for param in blocks.parameters():
                self.assertTrue(param.requires_grad)
            norm = getattr(model, f'norm{i + 1}')
            for param in norm.parameters():
                self.assertTrue(param.requires_grad)

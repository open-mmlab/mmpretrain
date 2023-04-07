# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile
from copy import deepcopy
from unittest import TestCase

import torch
from mmengine.runner import load_checkpoint, save_checkpoint
from torch import nn

from mmcls.models.backbones.focalnet import FocalModulationBlock, FocalNet


class TestFocalNet(TestCase):

    def setUp(self):
        self.cfg = dict(arch='t-srf', drop_path_rate=0.1)

    def test_arch(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            FocalNet(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'depths': [1, 1, 1, 1],
                'orders': [1, 1, 1, 1],
            }
            FocalNet(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        base_dim = 64
        depths = [2, 3, 18, 2]
        embed_dims = [base_dim * 2, base_dim * 4, base_dim * 8, base_dim * 8]
        cfg['arch'] = {
            'embed_dims': base_dim,
            'ffn_ratio': 4.,
            'depths': depths,
            'focal_levels': [2, 2, 2, 2],
            'focal_windows': [3, 3, 3, 3],
            'num_heads': [3, 6, 12, 24],
            'use_overlapped_embed': False,
            'use_postln': False,
            'use_layer_scale': False,
            'normalize_modulator': False,
            'use_postln_in_modulation': False,
        }
        model = FocalNet(**cfg)

        for i in range(len(depths)):
            stage = model.layers[i]
            self.assertEqual(stage.out_channels, embed_dims[i])
            self.assertEqual(len(stage.blocks), depths[i])

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
        model = FocalNet(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()

        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))

        tmpdir = tempfile.gettempdir()
        # Save checkpoints
        checkpoint = os.path.join(tmpdir, 'checkpoint.pth')
        save_checkpoint(model.state_dict(), checkpoint)

        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 96,
            'ffn_ratio': 4.,
            'depths': [2, 2, 6, 2],
            'focal_levels': [2, 2, 2, 2],
            'focal_windows': [9, 9, 9, 9],
            'num_heads': [3, 6, 12, 24],
            'use_overlapped_embed': False,
            'use_postln': False,
            'use_layer_scale': False,
            'normalize_modulator': False,
            'use_postln_in_modulation': False,
        }
        model2 = FocalNet(**cfg)
        load_checkpoint(model2, checkpoint, strict=False)

        weight1 = model.layers[0].blocks[0].modulation.focal_layers[
            0].conv.weight
        weight2 = model2.layers[0].blocks[0].modulation.focal_layers[
            0].conv.weight
        fsize1 = weight1.shape[2]
        fsize2 = weight2.shape[2]

        weight1_resized = torch.zeros(weight2.shape)
        weight1_resized[:, :, (fsize2 - fsize1) // 2:-(fsize2 - fsize1) // 2,
                        (fsize2 - fsize1) // 2:-(fsize2 - fsize1) //
                        2] = weight1
        self.assertFalse(torch.allclose(weight2, weight1_resized))

        os.remove(checkpoint)

    def test_forward(self):
        imgs = torch.randn(3, 3, 224, 224)

        cfg = deepcopy(self.cfg)
        model = FocalNet(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (3, 768, 7, 7))

        # test multiple output indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = (0, 1, 2, 3)
        model = FocalNet(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for emb_size, stride, out in zip([96, 192, 384, 768], [1, 2, 4, 8],
                                         outs):
            self.assertEqual(out.shape,
                             (3, emb_size, 56 // stride, 56 // stride))

        # test with checkpoint forward
        cfg = deepcopy(self.cfg)
        cfg['with_cp'] = True
        model = FocalNet(**cfg)
        for m in model.modules():
            if isinstance(m, FocalModulationBlock):
                self.assertTrue(m.with_cp)
        model.init_weights()
        model.train()

        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (3, 768, 7, 7))

        # test with large arch like
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 96,
            'ffn_ratio': 4.,
            'depths': [2, 2, 6, 2],
            'focal_levels': [2, 2, 2, 2],
            'focal_windows': [3, 3, 3, 3],
            'num_heads': [3, 6, 12, 24],
            'use_overlapped_embed': True,
            'use_postln': True,
            'use_layer_scale': True,
            'normalize_modulator': False,
            'use_postln_in_modulation': False,
        }
        model = FocalNet(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (3, 768, 9, 9))

        # test with large arch like with normalize_modulator
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 96,
            'ffn_ratio': 4.,
            'depths': [2, 2, 6, 2],
            'focal_levels': [2, 2, 2, 2],
            'focal_windows': [3, 3, 3, 3],
            'num_heads': [3, 6, 12, 24],
            'use_overlapped_embed': True,
            'use_postln': True,
            'use_layer_scale': True,
            'normalize_modulator': True,
            'use_postln_in_modulation': False,
        }
        model = FocalNet(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (3, 768, 9, 9))

        # test with huge arch like
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 96,
            'ffn_ratio': 4.,
            'depths': [2, 2, 6, 2],
            'focal_levels': [2, 2, 2, 2],
            'focal_windows': [3, 3, 3, 3],
            'num_heads': [3, 6, 12, 24],
            'use_overlapped_embed': True,
            'use_postln': True,
            'use_layer_scale': True,
            'normalize_modulator': False,
            'use_postln_in_modulation': True,
        }
        model = FocalNet(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (3, 768, 9, 9))

        # test with dynamic input shape
        imgs1 = torch.randn(3, 3, 224, 224)
        imgs2 = torch.randn(3, 3, 256, 256)
        imgs3 = torch.randn(3, 3, 256, 309)
        cfg = deepcopy(self.cfg)
        model = FocalNet(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            feat = outs[-1]
            expect_feat_shape = (math.ceil(imgs.shape[2] / 32),
                                 math.ceil(imgs.shape[3] / 32))
            self.assertEqual(feat.shape, (3, 768, *expect_feat_shape))

    def test_structure(self):
        # test drop_path_rate decay
        cfg = deepcopy(self.cfg)
        cfg['drop_path_rate'] = 0.2
        model = FocalNet(**cfg)
        depths = model.arch_settings['depths']
        stages = model.layers
        total_depth = sum(depths)
        dpr = [
            x.item()
            for x in torch.linspace(0, cfg['drop_path_rate'], total_depth)
        ]
        i = 0
        for stage in stages:
            for block in stage.blocks:
                expect_prob = dpr[i]
                if expect_prob == 0:
                    assert isinstance(block.drop_path, nn.Identity)
                else:
                    self.assertAlmostEqual(block.drop_path.drop_prob,
                                           expect_prob)
                i += 1

        # test with first stage frozen.
        cfg = deepcopy(self.cfg)
        frozen_stages = 0
        cfg['frozen_stages'] = frozen_stages
        cfg['out_indices'] = (0, 1, 2, 3)
        model = FocalNet(**cfg)
        model.init_weights()
        model.train()

        # the patch_embed and first stage should not require grad.
        self.assertFalse(model.patch_embed.training)
        for param in model.patch_embed.parameters():
            self.assertFalse(param.requires_grad)
        for i in range(frozen_stages + 1):
            stage = model.layers[i]
            for param in stage.parameters():
                self.assertFalse(param.requires_grad)
        for param in model.norm0.parameters():
            self.assertFalse(param.requires_grad)

        # the second stage should require grad.
        for i in range(frozen_stages + 1, 4):
            stage = model.layers[i]
            for param in stage.parameters():
                self.assertTrue(param.requires_grad)
            norm = getattr(model, f'norm{i}')
            for param in norm.parameters():
                self.assertTrue(param.requires_grad)

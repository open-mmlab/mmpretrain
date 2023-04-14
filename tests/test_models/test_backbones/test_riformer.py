# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch
import torch.nn as nn

from mmpretrain.models.backbones import RIFormer
from mmpretrain.models.backbones.riformer import RIFormerBlock


class TestRIFormer(TestCase):

    def setUp(self):
        arch = 's12'
        self.cfg = dict(arch=arch, drop_path_rate=0.1)
        self.arch = RIFormer.arch_settings[arch]

    def test_arch(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'Unavailable arch'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            RIFormer(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'must have "layers"'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'embed_dims': 96,
                'num_heads': [3, 6, 12, 16],
            }
            RIFormer(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        layers = [2, 2, 4, 2]
        embed_dims = [6, 12, 6, 12]
        mlp_ratios = [2, 3, 4, 4]
        layer_scale_init_value = 1e-4
        cfg['arch'] = dict(
            layers=layers,
            embed_dims=embed_dims,
            mlp_ratios=mlp_ratios,
            layer_scale_init_value=layer_scale_init_value,
        )
        model = RIFormer(**cfg)
        for i, stage in enumerate(model.network):
            if not isinstance(stage, RIFormerBlock):
                continue
            self.assertEqual(len(stage), layers[i])
            self.assertEqual(stage[0].mlp.fc1.in_channels, embed_dims[i])
            self.assertEqual(stage[0].mlp.fc1.out_channels,
                             embed_dims[i] * mlp_ratios[i])
            self.assertTrue(
                torch.allclose(stage[0].layer_scale_1,
                               torch.tensor(layer_scale_init_value)))
            self.assertTrue(
                torch.allclose(stage[0].layer_scale_2,
                               torch.tensor(layer_scale_init_value)))

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
        model = RIFormer(**cfg)
        ori_weight = model.patch_embed.proj.weight.clone().detach()

        model.init_weights()
        initialized_weight = model.patch_embed.proj.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)

        cfg = deepcopy(self.cfg)
        model = RIFormer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (1, 512, 7, 7))

        # test multiple output indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = (0, 2, 4, 6)
        model = RIFormer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for dim, stride, out in zip(self.arch['embed_dims'], [1, 2, 4, 8],
                                    outs):
            self.assertEqual(out.shape, (1, dim, 56 // stride, 56 // stride))

    def test_repameterization(self):
        # Test eval of "train" mode and "deploy" mode
        imgs = torch.randn(1, 3, 224, 224)
        gap = nn.AdaptiveAvgPool2d(output_size=(1))
        fc = nn.Linear(self.arch['embed_dims'][3], 10)

        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = (0, 2, 4, 6)
        model = RIFormer(**cfg)
        model.eval()
        feats = model(imgs)
        self.assertIsInstance(feats, tuple)
        feat = feats[-1]
        pred = fc(gap(feat).flatten(1))
        model.switch_to_deploy()
        for m in model.modules():
            if isinstance(m, RIFormerBlock):
                assert m.deploy is True
        feats_deploy = model(imgs)
        pred_deploy = fc(gap(feats_deploy[-1]).flatten(1))
        for i in range(4):
            torch.allclose(feats[i], feats_deploy[i])
        torch.allclose(pred, pred_deploy)

    def test_structure(self):
        # test drop_path_rate decay
        cfg = deepcopy(self.cfg)
        cfg['drop_path_rate'] = 0.2
        model = RIFormer(**cfg)
        layers = self.arch['layers']
        for i, block in enumerate(model.network):
            expect_prob = 0.2 / (sum(layers) - 1) * i
            if hasattr(block, 'drop_path'):
                if expect_prob == 0:
                    self.assertIsInstance(block.drop_path, torch.nn.Identity)
                else:
                    self.assertAlmostEqual(block.drop_path.drop_prob,
                                           expect_prob)

        # test with first stage frozen.
        cfg = deepcopy(self.cfg)
        frozen_stages = 1
        cfg['frozen_stages'] = frozen_stages
        cfg['out_indices'] = (0, 2, 4, 6)
        model = RIFormer(**cfg)
        model.init_weights()
        model.train()

        # the patch_embed and first stage should not require grad.
        self.assertFalse(model.patch_embed.training)
        for param in model.patch_embed.parameters():
            self.assertFalse(param.requires_grad)
        for i in range(frozen_stages):
            module = model.network[i]
            for param in module.parameters():
                self.assertFalse(param.requires_grad)
        for param in model.norm0.parameters():
            self.assertFalse(param.requires_grad)

        # the second stage should require grad.
        for i in range(frozen_stages + 1, 7):
            module = model.network[i]
            for param in module.parameters():
                self.assertTrue(param.requires_grad)
            if hasattr(model, f'norm{i}'):
                norm = getattr(model, f'norm{i}')
                for param in norm.parameters():
                    self.assertTrue(param.requires_grad)

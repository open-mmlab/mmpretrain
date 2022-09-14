# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch
from mmcv.cnn import ConvModule
from torch import nn

from mmcls.models.backbones import EfficientFormer
from mmcls.models.backbones.efficientformer import (AttentionWithBias, Flat,
                                                    Meta3D, Meta4D)
from mmcls.models.backbones.poolformer import Pooling


class TestEfficientFormer(TestCase):

    def setUp(self):
        self.cfg = dict(arch='l1', drop_path_rate=0.1)
        self.arch = EfficientFormer.arch_settings['l1']
        self.custom_arch = {
            'layers': [1, 1, 1, 4],
            'embed_dims': [48, 96, 224, 448],
            'downsamples': [False, True, True, True],
            'vit_num': 2,
        }
        self.custom_cfg = dict(arch=self.custom_arch)

    def test_arch(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'Unavailable arch'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            EfficientFormer(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'must have'):
            cfg = deepcopy(self.custom_cfg)
            cfg['arch'].pop('layers')
            EfficientFormer(**cfg)

        # Test vit_num < 0
        with self.assertRaisesRegex(AssertionError, "'vit_num' must"):
            cfg = deepcopy(self.custom_cfg)
            cfg['arch']['vit_num'] = -1
            EfficientFormer(**cfg)

        # Test vit_num > last stage layers
        with self.assertRaisesRegex(AssertionError, "'vit_num' must"):
            cfg = deepcopy(self.custom_cfg)
            cfg['arch']['vit_num'] = 10
            EfficientFormer(**cfg)

        # Test out_ind
        with self.assertRaisesRegex(AssertionError, '"out_indices" must'):
            cfg = deepcopy(self.custom_cfg)
            cfg['out_indices'] = dict
            EfficientFormer(**cfg)

        # Test custom arch
        cfg = deepcopy(self.custom_cfg)
        model = EfficientFormer(**cfg)
        self.assertEqual(len(model.patch_embed), 2)
        layers = self.custom_arch['layers']
        downsamples = self.custom_arch['downsamples']
        vit_num = self.custom_arch['vit_num']

        for i, stage in enumerate(model.network):
            if downsamples[i]:
                self.assertIsInstance(stage[0], ConvModule)
                self.assertEqual(stage[0].conv.stride, (2, 2))
                self.assertTrue(hasattr(stage[0].conv, 'bias'))
                self.assertTrue(isinstance(stage[0].bn, nn.BatchNorm2d))

            if i < len(model.network) - 1:
                self.assertIsInstance(stage[-1], Meta4D)
                self.assertIsInstance(stage[-1].token_mixer, Pooling)
                self.assertEqual(len(stage) - downsamples[i], layers[i])
            elif vit_num > 0:
                self.assertIsInstance(stage[-1], Meta3D)
                self.assertIsInstance(stage[-1].token_mixer, AttentionWithBias)
                self.assertEqual(len(stage) - downsamples[i] - 1, layers[i])
                flat_layer_idx = len(stage) - vit_num - downsamples[i]
                self.assertIsInstance(stage[flat_layer_idx], Flat)
                count = 0
                for layer in stage:
                    if isinstance(layer, Meta3D):
                        count += 1
                self.assertEqual(count, vit_num)

    def test_init_weights(self):
        # test weight init cfg
        cfg = deepcopy(self.cfg)
        cfg['init_cfg'] = [
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear'),
            dict(type='Constant', layer=['LayerScale'], val=1e-4)
        ]
        model = EfficientFormer(**cfg)
        ori_weight = model.patch_embed[0].conv.weight.clone().detach()
        ori_ls_weight = model.network[0][-1].ls1.weight.clone().detach()

        model.init_weights()
        initialized_weight = model.patch_embed[0].conv.weight
        initialized_ls_weight = model.network[0][-1].ls1.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))
        self.assertFalse(torch.allclose(ori_ls_weight, initialized_ls_weight))

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)

        # test last stage output
        cfg = deepcopy(self.cfg)
        model = EfficientFormer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (1, 448, 49))
        assert hasattr(model, 'norm3')
        assert isinstance(getattr(model, 'norm3'), nn.LayerNorm)

        # test multiple output indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = (0, 1, 2, 3)
        cfg['reshape_last_feat'] = True
        model = EfficientFormer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        # Test out features shape
        for dim, stride, out in zip(self.arch['embed_dims'], [1, 2, 4, 8],
                                    outs):
            self.assertEqual(out.shape, (1, dim, 56 // stride, 56 // stride))

        # Test norm layer
        for i in range(4):
            assert hasattr(model, f'norm{i}')
            stage_norm = getattr(model, f'norm{i}')
            assert isinstance(stage_norm, nn.GroupNorm)
            assert stage_norm.num_groups == 1

        # Test vit_num == 0
        cfg = deepcopy(self.custom_cfg)
        cfg['arch']['vit_num'] = 0
        cfg['out_indices'] = (0, 1, 2, 3)
        model = EfficientFormer(**cfg)
        for i in range(4):
            assert hasattr(model, f'norm{i}')
            stage_norm = getattr(model, f'norm{i}')
            assert isinstance(stage_norm, nn.GroupNorm)
            assert stage_norm.num_groups == 1

    def test_structure(self):
        # test drop_path_rate decay
        cfg = deepcopy(self.cfg)
        cfg['drop_path_rate'] = 0.2
        model = EfficientFormer(**cfg)
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
        cfg['out_indices'] = (0, 1, 2, 3)
        model = EfficientFormer(**cfg)
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
        for i in range(frozen_stages + 1, 4):
            module = model.network[i]
            for param in module.parameters():
                self.assertTrue(param.requires_grad)
            if hasattr(model, f'norm{i}'):
                norm = getattr(model, f'norm{i}')
                for param in norm.parameters():
                    self.assertTrue(param.requires_grad)

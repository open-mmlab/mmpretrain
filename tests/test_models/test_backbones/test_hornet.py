# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from itertools import chain
from unittest import TestCase

import pytest
import torch
from mmengine.utils import digit_version
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch import nn

from mmpretrain.models.backbones import HorNet


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


@pytest.mark.skipif(
    digit_version(torch.__version__) < digit_version('1.7.0'),
    reason='torch.fft is not available before 1.7.0')
class TestHorNet(TestCase):

    def setUp(self):
        self.cfg = dict(
            arch='t', drop_path_rate=0.1, gap_before_final_norm=False)

    def test_arch(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            HorNet(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'depths': [1, 1, 1, 1],
                'orders': [1, 1, 1, 1],
            }
            HorNet(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        base_dim = 64
        depths = [2, 3, 18, 2]
        embed_dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        cfg['arch'] = {
            'base_dim':
            base_dim,
            'depths':
            depths,
            'orders': [2, 3, 4, 5],
            'dw_cfg': [
                dict(type='DW', kernel_size=7),
                dict(type='DW', kernel_size=7),
                dict(type='GF', h=14, w=8),
                dict(type='GF', h=7, w=4)
            ],
        }
        model = HorNet(**cfg)

        for i in range(len(depths)):
            stage = model.stages[i]
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
        model = HorNet(**cfg)
        ori_weight = model.downsample_layers[0][0].weight.clone().detach()

        model.init_weights()
        initialized_weight = model.downsample_layers[0][0].weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))

    def test_forward(self):
        imgs = torch.randn(3, 3, 224, 224)

        cfg = deepcopy(self.cfg)
        model = HorNet(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (3, 512, 7, 7))

        # test multiple output indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = (0, 1, 2, 3)
        model = HorNet(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for emb_size, stride, out in zip([64, 128, 256, 512], [1, 2, 4, 8],
                                         outs):
            self.assertEqual(out.shape,
                             (3, emb_size, 56 // stride, 56 // stride))

        # test with dynamic input shape
        imgs1 = torch.randn(3, 3, 224, 224)
        imgs2 = torch.randn(3, 3, 256, 256)
        imgs3 = torch.randn(3, 3, 256, 309)
        cfg = deepcopy(self.cfg)
        model = HorNet(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            feat = outs[-1]
            expect_feat_shape = (math.floor(imgs.shape[2] / 32),
                                 math.floor(imgs.shape[3] / 32))
            self.assertEqual(feat.shape, (3, 512, *expect_feat_shape))

    def test_structure(self):
        # test drop_path_rate decay
        cfg = deepcopy(self.cfg)
        cfg['drop_path_rate'] = 0.2
        model = HorNet(**cfg)
        depths = model.arch_settings['depths']
        stages = model.stages
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

        # test VAN with first stage frozen.
        cfg = deepcopy(self.cfg)
        frozen_stages = 0
        cfg['frozen_stages'] = frozen_stages
        cfg['out_indices'] = (0, 1, 2, 3)
        model = HorNet(**cfg)
        model.init_weights()
        model.train()

        # the patch_embed and first stage should not require grad.
        for i in range(frozen_stages + 1):
            down = model.downsample_layers[i]
            for param in down.parameters():
                self.assertFalse(param.requires_grad)
            blocks = model.stages[i]
            for param in blocks.parameters():
                self.assertFalse(param.requires_grad)

        # the second stage should require grad.
        for i in range(frozen_stages + 1, 4):
            down = model.downsample_layers[i]
            for param in down.parameters():
                self.assertTrue(param.requires_grad)
            blocks = model.stages[i]
            for param in blocks.parameters():
                self.assertTrue(param.requires_grad)

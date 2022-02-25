# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile
from copy import deepcopy
from unittest import TestCase

import torch
from mmcv.runner import load_checkpoint, save_checkpoint

from mmcls.models.backbones import DistilledVisionTransformer
from .utils import timm_resize_pos_embed


class TestDeiT(TestCase):

    def setUp(self):
        self.cfg = dict(
            arch='deit-base', img_size=224, patch_size=16, drop_rate=0.1)

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
        model = DistilledVisionTransformer(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()
        # The pos_embed is all zero before initialize
        self.assertTrue(torch.allclose(model.dist_token, torch.tensor(0.)))

        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))
        self.assertFalse(torch.allclose(model.dist_token, torch.tensor(0.)))

        # test load checkpoint
        pretrain_pos_embed = model.pos_embed.clone().detach()
        tmpdir = tempfile.gettempdir()
        checkpoint = os.path.join(tmpdir, 'test.pth')
        save_checkpoint(model, checkpoint)
        cfg = deepcopy(self.cfg)
        model = DistilledVisionTransformer(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        self.assertTrue(torch.allclose(model.pos_embed, pretrain_pos_embed))

        # test load checkpoint with different img_size
        cfg = deepcopy(self.cfg)
        cfg['img_size'] = 384
        model = DistilledVisionTransformer(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        resized_pos_embed = timm_resize_pos_embed(
            pretrain_pos_embed, model.pos_embed, num_tokens=2)
        self.assertTrue(torch.allclose(model.pos_embed, resized_pos_embed))

        os.remove(checkpoint)

    def test_forward(self):
        imgs = torch.randn(3, 3, 224, 224)

        # test with_cls_token=False
        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['output_cls_token'] = True
        with self.assertRaisesRegex(AssertionError, 'but got False'):
            DistilledVisionTransformer(**cfg)

        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['output_cls_token'] = False
        model = DistilledVisionTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (3, 768, 14, 14))

        # test with output_cls_token
        cfg = deepcopy(self.cfg)
        model = DistilledVisionTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token, cls_token, dist_token = outs[-1]
        self.assertEqual(patch_token.shape, (3, 768, 14, 14))
        self.assertEqual(cls_token.shape, (3, 768))
        self.assertEqual(dist_token.shape, (3, 768))

        # test without output_cls_token
        cfg = deepcopy(self.cfg)
        cfg['output_cls_token'] = False
        model = DistilledVisionTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (3, 768, 14, 14))

        # Test forward with multi out indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = [-3, -2, -1]
        model = DistilledVisionTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 3)
        for out in outs:
            patch_token, cls_token, dist_token = out
            self.assertEqual(patch_token.shape, (3, 768, 14, 14))
            self.assertEqual(cls_token.shape, (3, 768))
            self.assertEqual(dist_token.shape, (3, 768))

        # Test forward with dynamic input size
        imgs1 = torch.randn(3, 3, 224, 224)
        imgs2 = torch.randn(3, 3, 256, 256)
        imgs3 = torch.randn(3, 3, 256, 309)
        cfg = deepcopy(self.cfg)
        model = DistilledVisionTransformer(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            patch_token, cls_token, dist_token = outs[-1]
            expect_feat_shape = (math.ceil(imgs.shape[2] / 16),
                                 math.ceil(imgs.shape[3] / 16))
            self.assertEqual(patch_token.shape, (3, 768, *expect_feat_shape))
            self.assertEqual(cls_token.shape, (3, 768))
            self.assertEqual(dist_token.shape, (3, 768))

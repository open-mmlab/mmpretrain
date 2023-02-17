# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile
from copy import deepcopy
from unittest import TestCase

import torch
from mmengine.runner import load_checkpoint, save_checkpoint

from mmpretrain.models.backbones import T2T_ViT
from .utils import timm_resize_pos_embed


class TestT2TViT(TestCase):

    def setUp(self):
        self.cfg = dict(
            img_size=224,
            in_channels=3,
            embed_dims=384,
            t2t_cfg=dict(
                token_dims=64,
                use_performer=False,
            ),
            num_layers=14,
            drop_path_rate=0.1)

    def test_structure(self):
        # The performer hasn't been implemented
        cfg = deepcopy(self.cfg)
        cfg['t2t_cfg']['use_performer'] = True
        with self.assertRaises(NotImplementedError):
            T2T_ViT(**cfg)

        # Test out_indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = {1: 1}
        with self.assertRaisesRegex(AssertionError, "get <class 'dict'>"):
            T2T_ViT(**cfg)
        cfg['out_indices'] = [0, 15]
        with self.assertRaisesRegex(AssertionError, 'Invalid out_indices 15'):
            T2T_ViT(**cfg)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = T2T_ViT(**cfg)
        self.assertEqual(len(model.encoder), 14)
        dpr_inc = 0.1 / (14 - 1)
        dpr = 0
        for layer in model.encoder:
            self.assertEqual(layer.attn.embed_dims, 384)
            # The default mlp_ratio is 3
            self.assertEqual(layer.ffn.feedforward_channels, 384 * 3)
            self.assertAlmostEqual(layer.attn.out_drop.drop_prob, dpr)
            self.assertAlmostEqual(layer.ffn.dropout_layer.drop_prob, dpr)
            dpr += dpr_inc

    def test_init_weights(self):
        # test weight init cfg
        cfg = deepcopy(self.cfg)
        cfg['init_cfg'] = [dict(type='TruncNormal', layer='Linear', std=.02)]
        model = T2T_ViT(**cfg)
        ori_weight = model.tokens_to_token.project.weight.clone().detach()

        model.init_weights()
        initialized_weight = model.tokens_to_token.project.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))

        # test load checkpoint
        pretrain_pos_embed = model.pos_embed.clone().detach()
        tmpdir = tempfile.gettempdir()
        checkpoint = os.path.join(tmpdir, 'test.pth')
        save_checkpoint(model.state_dict(), checkpoint)
        cfg = deepcopy(self.cfg)
        model = T2T_ViT(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        self.assertTrue(torch.allclose(model.pos_embed, pretrain_pos_embed))

        # test load checkpoint with different img_size
        cfg = deepcopy(self.cfg)
        cfg['img_size'] = 384
        model = T2T_ViT(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        resized_pos_embed = timm_resize_pos_embed(pretrain_pos_embed,
                                                  model.pos_embed)
        self.assertTrue(torch.allclose(model.pos_embed, resized_pos_embed))

        os.remove(checkpoint)

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)

        # test with_cls_token=False
        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['output_cls_token'] = True
        with self.assertRaisesRegex(AssertionError, 'but got False'):
            T2T_ViT(**cfg)

        cfg = deepcopy(self.cfg)
        cfg['with_cls_token'] = False
        cfg['output_cls_token'] = False
        model = T2T_ViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 384, 14, 14))

        # test with output_cls_token
        cfg = deepcopy(self.cfg)
        model = T2T_ViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token, cls_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 384, 14, 14))
        self.assertEqual(cls_token.shape, (1, 384))

        # test without output_cls_token
        cfg = deepcopy(self.cfg)
        cfg['output_cls_token'] = False
        model = T2T_ViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 384, 14, 14))

        # Test forward with multi out indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = [-3, -2, -1]
        model = T2T_ViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 3)
        for out in outs:
            patch_token, cls_token = out
            self.assertEqual(patch_token.shape, (1, 384, 14, 14))
            self.assertEqual(cls_token.shape, (1, 384))

        # Test forward with dynamic input size
        imgs1 = torch.randn(1, 3, 224, 224)
        imgs2 = torch.randn(1, 3, 256, 256)
        imgs3 = torch.randn(1, 3, 256, 309)
        cfg = deepcopy(self.cfg)
        model = T2T_ViT(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            patch_token, cls_token = outs[-1]
            expect_feat_shape = (math.ceil(imgs.shape[2] / 16),
                                 math.ceil(imgs.shape[3] / 16))
            self.assertEqual(patch_token.shape, (1, 384, *expect_feat_shape))
            self.assertEqual(cls_token.shape, (1, 384))

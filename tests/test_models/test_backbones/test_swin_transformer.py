# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile
from copy import deepcopy
from itertools import chain
from unittest import TestCase

import torch
from mmcv.runner import load_checkpoint, save_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmcls.models.backbones import SwinTransformer
from mmcls.models.backbones.swin_transformer import SwinBlock
from .utils import timm_resize_pos_embed


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


class TestSwinTransformer(TestCase):

    def setUp(self):
        self.cfg = dict(
            arch='b', img_size=224, patch_size=4, drop_path_rate=0.1)

    def test_arch(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            SwinTransformer(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'embed_dims': 96,
                'num_heads': [3, 6, 12, 16],
            }
            SwinTransformer(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        depths = [2, 2, 4, 2]
        num_heads = [6, 12, 6, 12]
        cfg['arch'] = {
            'embed_dims': 256,
            'depths': depths,
            'num_heads': num_heads
        }
        model = SwinTransformer(**cfg)
        for i, stage in enumerate(model.stages):
            self.assertEqual(stage.embed_dims, 256 * (2**i))
            self.assertEqual(len(stage.blocks), depths[i])
            self.assertEqual(stage.blocks[0].attn.w_msa.num_heads,
                             num_heads[i])

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
        model = SwinTransformer(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()
        # The pos_embed is all zero before initialize
        self.assertTrue(
            torch.allclose(model.absolute_pos_embed, torch.tensor(0.)))

        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))
        self.assertFalse(
            torch.allclose(model.absolute_pos_embed, torch.tensor(0.)))

        pretrain_pos_embed = model.absolute_pos_embed.clone().detach()

        tmpdir = tempfile.gettempdir()
        # Save v3 checkpoints
        checkpoint_v2 = os.path.join(tmpdir, 'v3.pth')
        save_checkpoint(model, checkpoint_v2)
        # Save v1 checkpoints
        setattr(model, 'norm', model.norm3)
        setattr(model.stages[0].blocks[1].attn, 'attn_mask',
                torch.zeros(64, 49, 49))
        model._version = 1
        del model.norm3
        checkpoint_v1 = os.path.join(tmpdir, 'v1.pth')
        save_checkpoint(model, checkpoint_v1)

        # test load v1 checkpoint
        cfg = deepcopy(self.cfg)
        cfg['use_abs_pos_embed'] = True
        model = SwinTransformer(**cfg)
        load_checkpoint(model, checkpoint_v1, strict=True)

        # test load v3 checkpoint
        cfg = deepcopy(self.cfg)
        cfg['use_abs_pos_embed'] = True
        model = SwinTransformer(**cfg)
        load_checkpoint(model, checkpoint_v2, strict=True)

        # test load v3 checkpoint with different img_size
        cfg = deepcopy(self.cfg)
        cfg['img_size'] = 384
        cfg['use_abs_pos_embed'] = True
        model = SwinTransformer(**cfg)
        load_checkpoint(model, checkpoint_v2, strict=True)
        resized_pos_embed = timm_resize_pos_embed(
            pretrain_pos_embed, model.absolute_pos_embed, num_tokens=0)
        self.assertTrue(
            torch.allclose(model.absolute_pos_embed, resized_pos_embed))

        os.remove(checkpoint_v1)
        os.remove(checkpoint_v2)

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)

        cfg = deepcopy(self.cfg)
        model = SwinTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (1, 1024, 7, 7))

        # test with window_size=12
        cfg = deepcopy(self.cfg)
        cfg['window_size'] = 12
        model = SwinTransformer(**cfg)
        outs = model(torch.randn(1, 3, 384, 384))
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (1, 1024, 12, 12))
        with self.assertRaisesRegex(AssertionError, r'the window size \(12\)'):
            model(torch.randn(1, 3, 224, 224))

        # test with pad_small_map=True
        cfg = deepcopy(self.cfg)
        cfg['window_size'] = 12
        cfg['pad_small_map'] = True
        model = SwinTransformer(**cfg)
        outs = model(torch.randn(1, 3, 224, 224))
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (1, 1024, 7, 7))

        # test multiple output indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = (0, 1, 2, 3)
        model = SwinTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for stride, out in zip([1, 2, 4, 8], outs):
            self.assertEqual(out.shape,
                             (1, 128 * stride, 56 // stride, 56 // stride))

        # test with checkpoint forward
        cfg = deepcopy(self.cfg)
        cfg['with_cp'] = True
        model = SwinTransformer(**cfg)
        for m in model.modules():
            if isinstance(m, SwinBlock):
                self.assertTrue(m.with_cp)
        model.init_weights()
        model.train()

        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        feat = outs[-1]
        self.assertEqual(feat.shape, (1, 1024, 7, 7))

        # test with dynamic input shape
        imgs1 = torch.randn(1, 3, 224, 224)
        imgs2 = torch.randn(1, 3, 256, 256)
        imgs3 = torch.randn(1, 3, 256, 309)
        cfg = deepcopy(self.cfg)
        model = SwinTransformer(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            feat = outs[-1]
            expect_feat_shape = (math.ceil(imgs.shape[2] / 32),
                                 math.ceil(imgs.shape[3] / 32))
            self.assertEqual(feat.shape, (1, 1024, *expect_feat_shape))

    def test_structure(self):
        # test drop_path_rate decay
        cfg = deepcopy(self.cfg)
        cfg['drop_path_rate'] = 0.2
        model = SwinTransformer(**cfg)
        depths = model.arch_settings['depths']
        blocks = chain(*[stage.blocks for stage in model.stages])
        for i, block in enumerate(blocks):
            expect_prob = 0.2 / (sum(depths) - 1) * i
            self.assertAlmostEqual(block.ffn.dropout_layer.drop_prob,
                                   expect_prob)
            self.assertAlmostEqual(block.attn.drop.drop_prob, expect_prob)

        # test Swin-Transformer with norm_eval=True
        cfg = deepcopy(self.cfg)
        cfg['norm_eval'] = True
        cfg['norm_cfg'] = dict(type='BN')
        cfg['stage_cfgs'] = dict(block_cfgs=dict(norm_cfg=dict(type='BN')))
        model = SwinTransformer(**cfg)
        model.init_weights()
        model.train()
        self.assertTrue(check_norm_state(model.modules(), False))

        # test Swin-Transformer with first stage frozen.
        cfg = deepcopy(self.cfg)
        frozen_stages = 0
        cfg['frozen_stages'] = frozen_stages
        cfg['out_indices'] = (0, 1, 2, 3)
        model = SwinTransformer(**cfg)
        model.init_weights()
        model.train()

        # the patch_embed and first stage should not require grad.
        self.assertFalse(model.patch_embed.training)
        for param in model.patch_embed.parameters():
            self.assertFalse(param.requires_grad)
        for i in range(frozen_stages + 1):
            stage = model.stages[i]
            for param in stage.parameters():
                self.assertFalse(param.requires_grad)
        for param in model.norm0.parameters():
            self.assertFalse(param.requires_grad)

        # the second stage should require grad.
        for i in range(frozen_stages + 1, 4):
            stage = model.stages[i]
            for param in stage.parameters():
                self.assertTrue(param.requires_grad)
            norm = getattr(model, f'norm{i}')
            for param in norm.parameters():
                self.assertTrue(param.requires_grad)

# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from copy import deepcopy
from unittest import TestCase

import torch
from mmcv.runner import load_checkpoint, save_checkpoint

from mmcls.models.backbones import RepMLPNet


class TestRepMLP(TestCase):

    def setUp(self):
        # default model setting
        self.cfg = dict(
            arch='b',
            img_size=224,
            out_indices=(3, ),
            reparam_conv_kernels=(1, 3),
            final_norm=True)

        # default model setting and output stage channels
        self.model_forward_settings = [
            dict(model_name='B', out_sizes=(96, 192, 384, 768)),
        ]

        # temp ckpt path
        self.ckpt_path = os.path.join(tempfile.gettempdir(), 'ckpt.pth')

    def test_arch(self):
        # Test invalid arch data type
        with self.assertRaisesRegex(AssertionError, 'arch needs a dict'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = [96, 192, 384, 768]
            RepMLPNet(**cfg)

        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'A'
            RepMLPNet(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'channels': [96, 192, 384, 768],
                'depths': [2, 2, 12, 2]
            }
            RepMLPNet(**cfg)

        # test len(arch['depths']) equals to len(arch['channels'])
        # equals to len(arch['sharesets_nums'])
        with self.assertRaisesRegex(AssertionError, 'Length of setting'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'channels': [96, 192, 384, 768],
                'depths': [2, 2, 12, 2],
                'sharesets_nums': [1, 4, 32]
            }
            RepMLPNet(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        channels = [96, 192, 384, 768]
        depths = [2, 2, 12, 2]
        sharesets_nums = [1, 4, 32, 128]
        cfg['arch'] = {
            'channels': channels,
            'depths': depths,
            'sharesets_nums': sharesets_nums
        }
        cfg['out_indices'] = (0, 1, 2, 3)
        model = RepMLPNet(**cfg)
        for i, stage in enumerate(model.stages):
            self.assertEqual(len(stage), depths[i])
            self.assertEqual(stage[0].repmlp_block.channels, channels[i])
            self.assertEqual(stage[0].repmlp_block.deploy, False)
            self.assertEqual(stage[0].repmlp_block.num_sharesets,
                             sharesets_nums[i])

    def test_init(self):
        # test weight init cfg
        cfg = deepcopy(self.cfg)
        cfg['init_cfg'] = [
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]
        model = RepMLPNet(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()

        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)
        cfg = deepcopy(self.cfg)
        model = RepMLPNet(**cfg)
        feat = model(imgs)
        self.assertTrue(isinstance(feat, tuple))
        self.assertEqual(len(feat), 1)
        self.assertTrue(isinstance(feat[0], torch.Tensor))
        self.assertEqual(feat[0].shape, torch.Size((1, 768, 7, 7)))

        imgs = torch.randn(1, 3, 256, 256)
        with self.assertRaisesRegex(AssertionError, "doesn't support dynamic"):
            model(imgs)

        # Test RepMLPNet model forward
        for model_test_setting in self.model_forward_settings:
            model = RepMLPNet(
                model_test_setting['model_name'],
                out_indices=(0, 1, 2, 3),
                final_norm=False)
            model.init_weights()

            model.train()
            imgs = torch.randn(1, 3, 224, 224)
            feat = model(imgs)
            self.assertEqual(
                feat[0].shape,
                torch.Size((1, model_test_setting['out_sizes'][1], 28, 28)))
            self.assertEqual(
                feat[1].shape,
                torch.Size((1, model_test_setting['out_sizes'][2], 14, 14)))
            self.assertEqual(
                feat[2].shape,
                torch.Size((1, model_test_setting['out_sizes'][3], 7, 7)))
            self.assertEqual(
                feat[3].shape,
                torch.Size((1, model_test_setting['out_sizes'][3], 7, 7)))

    def test_deploy_(self):
        # Test output before and load from deploy checkpoint
        imgs = torch.randn((1, 3, 224, 224))
        cfg = dict(
            arch='b', out_indices=(
                1,
                3,
            ), reparam_conv_kernels=(1, 3, 5))
        model = RepMLPNet(**cfg)

        model.eval()
        feats = model(imgs)
        model.switch_to_deploy()
        for m in model.modules():
            if hasattr(m, 'deploy'):
                self.assertTrue(m.deploy)
        model.eval()
        feats_ = model(imgs)
        assert len(feats) == len(feats_)
        for i in range(len(feats)):
            self.assertTrue(
                torch.allclose(
                    feats[i].sum(), feats_[i].sum(), rtol=0.1, atol=0.1))

        cfg['deploy'] = True
        model_deploy = RepMLPNet(**cfg)
        model_deploy.eval()
        save_checkpoint(model, self.ckpt_path)
        load_checkpoint(model_deploy, self.ckpt_path, strict=True)
        feats__ = model_deploy(imgs)

        assert len(feats_) == len(feats__)
        for i in range(len(feats)):
            self.assertTrue(torch.allclose(feats__[i], feats_[i]))

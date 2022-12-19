# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import tempfile
from copy import deepcopy
from unittest import TestCase

import torch
from mmengine.runner import load_checkpoint, save_checkpoint

from mmcls.models.backbones import GCViT


class TestGCViT(TestCase):

    def setUp(self):
        self.cfg = dict(
            arch='xxtiny')

    def test_structure(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            GCViT(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'num_heads': [2, 4, 8, 16],
                'dim': 64,
                'mlp_ratio': 3,
                'drop_path_rate': 0.2,
                'layer_scale': None
            }
            GCViT(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'depths': [2, 2, 6, 2],
            'num_heads': [2, 4, 8, 16],
            'dim': 64,
            'mlp_ratio': 3,
            'drop_path_rate': 0.2,
            'layer_scale': None
        }
        model = GCViT(**cfg)
        self.assertEqual(model.arch_settings['depths'], [2, 2, 6, 2])
        self.assertEqual(model.arch_settings['num_heads'],
                             [2, 4, 8, 16])
        self.assertEqual(model.arch_settings['dim'], 64)
        self.assertEqual(model.arch_settings['mlp_ratio'], 3)
        self.assertEqual(model.arch_settings['drop_path_rate'], 0.2)
        self.assertEqual(model.arch_settings['layer_scale'], None)

        # Test out_indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = {1: 1}
        with self.assertRaisesRegex(AssertionError, "get <class 'dict'>"):
            GCViT(**cfg)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = GCViT(**cfg)
        self.assertEqual(len(model.levels), 4)
    
    def test_init_weights(self):
        # test weight init cfg
        cfg = deepcopy(self.cfg)
        model = GCViT(**cfg)
        ori_weight = model.patch_embed.proj.weight.clone().detach()

        model.init_weights()
        initialized_weight = model.patch_embed.proj.weight
        self.assertTrue(torch.allclose(ori_weight, 
                        initialized_weight, 
                        rtol=1e-5,
                        ))

        # test load checkpoint
        pretrain_patch_embed = model.patch_embed.proj.weight.clone().detach()
        tmpdir = tempfile.gettempdir()
        checkpoint = os.path.join(tmpdir, 'test.pth')
        save_checkpoint(model.state_dict(), checkpoint)
        cfg = deepcopy(self.cfg)
        model = GCViT(**cfg)
        load_checkpoint(model, checkpoint, strict=True)
        self.assertTrue(torch.allclose(model.patch_embed.proj.weight,
              pretrain_patch_embed))

        # test load checkpoint with different img_size
        cfg = deepcopy(self.cfg)
        cfg['img_size'] = 384
        model = GCViT(**cfg)
        load_checkpoint(model, checkpoint, strict=True)

        os.remove(checkpoint)
    
    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)
        cfg = deepcopy(self.cfg)
        model = GCViT(**cfg)

        feat = model(imgs)
        assert len(feat) == 1
        assert feat[0].shape == torch.Size([1, 512])
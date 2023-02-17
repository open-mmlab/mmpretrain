# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial
from unittest import TestCase

import torch
from mmcv.cnn import ConvModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpretrain.models.backbones import CSPDarkNet, CSPResNet, CSPResNeXt
from mmpretrain.models.backbones.cspnet import (CSPNet, DarknetBottleneck,
                                                ResNetBottleneck,
                                                ResNeXtBottleneck)


class TestCSPNet(TestCase):

    def setUp(self):
        self.arch = dict(
            block_fn=(DarknetBottleneck, ResNetBottleneck, ResNeXtBottleneck),
            in_channels=(32, 64, 128),
            out_channels=(64, 128, 256),
            num_blocks=(1, 2, 8),
            expand_ratio=(2, 1, 1),
            bottle_ratio=(3, 1, 1),
            has_downsampler=True,
            down_growth=True,
            block_args=({}, {}, dict(base_channels=32)))
        self.stem_fn = partial(torch.nn.Conv2d, out_channels=32, kernel_size=3)

    def test_structure(self):
        # Test with attribute arch_setting.
        model = CSPNet(arch=self.arch, stem_fn=self.stem_fn, out_indices=[-1])
        self.assertEqual(len(model.stages), 3)
        self.assertEqual(type(model.stages[0].blocks[0]), DarknetBottleneck)
        self.assertEqual(type(model.stages[1].blocks[0]), ResNetBottleneck)
        self.assertEqual(type(model.stages[2].blocks[0]), ResNeXtBottleneck)


class TestCSPDarkNet(TestCase):

    def setUp(self):
        self.class_name = CSPDarkNet
        self.cfg = dict(depth=53)
        self.out_channels = [64, 128, 256, 512, 1024]
        self.all_out_indices = [0, 1, 2, 3, 4]
        self.frozen_stages = 2
        self.stem_down = (1, 1)
        self.num_stages = 5

    def test_structure(self):
        # Test invalid default depths
        with self.assertRaisesRegex(AssertionError, 'depth must be one of'):
            cfg = deepcopy(self.cfg)
            cfg['depth'] = 'unknown'
            self.class_name(**cfg)

        # Test out_indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = {1: 1}
        with self.assertRaisesRegex(AssertionError, "get <class 'dict'>"):
            self.class_name(**cfg)
        cfg['out_indices'] = [0, 13]
        with self.assertRaisesRegex(AssertionError, 'Invalid out_indices 13'):
            self.class_name(**cfg)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = self.class_name(**cfg)
        self.assertEqual(len(model.stages), self.num_stages)

    def test_forward(self):
        imgs = torch.randn(3, 3, 224, 224)

        # test without output_cls_token
        cfg = deepcopy(self.cfg)
        model = self.class_name(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[-1].size(), (3, self.out_channels[-1], 7, 7))

        # Test forward with multi out indices
        cfg = deepcopy(self.cfg)
        cfg['out_indices'] = self.all_out_indices
        model = self.class_name(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), len(self.all_out_indices))
        w, h = 224 / self.stem_down[0], 224 / self.stem_down[1]
        for i, out in enumerate(outs):
            self.assertEqual(
                out.size(),
                (3, self.out_channels[i], w // 2**(i + 1), h // 2**(i + 1)))

        # Test frozen stages
        cfg = deepcopy(self.cfg)
        cfg['frozen_stages'] = self.frozen_stages
        model = self.class_name(**cfg)
        model.init_weights()
        model.train()
        assert model.stem.training is False
        for param in model.stem.parameters():
            assert param.requires_grad is False
        for i in range(self.frozen_stages + 1):
            stage = model.stages[i]
            for mod in stage.modules():
                if isinstance(mod, _BatchNorm):
                    assert mod.training is False, i
            for param in stage.parameters():
                assert param.requires_grad is False


class TestCSPResNet(TestCSPDarkNet):

    def setUp(self):
        self.class_name = CSPResNet
        self.cfg = dict(depth=50)
        self.out_channels = [128, 256, 512, 1024]
        self.all_out_indices = [0, 1, 2, 3]
        self.frozen_stages = 2
        self.stem_down = (2, 2)
        self.num_stages = 4

    def test_deep_stem(self, ):
        cfg = deepcopy(self.cfg)
        cfg['deep_stem'] = True
        model = self.class_name(**cfg)
        self.assertEqual(len(model.stem), 3)
        for i in range(3):
            self.assertEqual(type(model.stem[i]), ConvModule)


class TestCSPResNeXt(TestCSPDarkNet):

    def setUp(self):
        self.class_name = CSPResNeXt
        self.cfg = dict(depth=50)
        self.out_channels = [256, 512, 1024, 2048]
        self.all_out_indices = [0, 1, 2, 3]
        self.frozen_stages = 2
        self.stem_down = (2, 2)
        self.num_stages = 4


if __name__ == '__main__':
    import unittest
    unittest.main()

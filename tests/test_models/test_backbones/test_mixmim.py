# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch

from mmpretrain.models.backbones import MixMIMTransformer


class TestMixMIM(TestCase):

    def setUp(self):
        self.cfg = dict(arch='b', drop_rate=0.0, drop_path_rate=0.1)

    def test_structure(self):

        # Test custom arch
        cfg = deepcopy(self.cfg)

        model = MixMIMTransformer(**cfg)
        self.assertEqual(model.embed_dims, 128)
        self.assertEqual(sum(model.depths), 24)
        self.assertIsNotNone(model.absolute_pos_embed)

        num_heads = [4, 8, 16, 32]
        for i, layer in enumerate(model.layers):
            self.assertEqual(layer.blocks[0].num_heads, num_heads[i])
            self.assertEqual(layer.blocks[0].ffn.feedforward_channels,
                             128 * (2**i) * 4)

    def test_forward(self):
        imgs = torch.randn(1, 3, 224, 224)

        cfg = deepcopy(self.cfg)
        model = MixMIMTransformer(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        averaged_token = outs[-1]
        self.assertEqual(averaged_token.shape, (1, 1024))

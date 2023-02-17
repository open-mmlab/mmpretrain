# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpretrain.models.utils import LayerScale


class TestLayerScale(TestCase):

    def test_init(self):
        with self.assertRaisesRegex(AssertionError, "'data_format' could"):
            cfg = dict(
                dim=10,
                data_format='BNC',
            )
            LayerScale(**cfg)

        cfg = dict(dim=10)
        ls = LayerScale(**cfg)
        assert torch.equal(ls.weight,
                           torch.ones(10, requires_grad=True) * 1e-5)

    def forward(self):
        # Test channels_last
        cfg = dict(dim=256, inplace=False, data_format='channels_last')
        ls_channels_last = LayerScale(**cfg)
        x = torch.randn((4, 49, 256))
        out = ls_channels_last(x)
        self.assertEqual(tuple(out.size()), (4, 49, 256))
        assert torch.equal(x * 1e-5, out)

        # Test channels_first
        cfg = dict(dim=256, inplace=False, data_format='channels_first')
        ls_channels_first = LayerScale(**cfg)
        x = torch.randn((4, 256, 7, 7))
        out = ls_channels_first(x)
        self.assertEqual(tuple(out.size()), (4, 256, 7, 7))
        assert torch.equal(x * 1e-5, out)

        # Test inplace True
        cfg = dict(dim=256, inplace=True, data_format='channels_first')
        ls_channels_first = LayerScale(**cfg)
        x = torch.randn((4, 256, 7, 7))
        out = ls_channels_first(x)
        self.assertEqual(tuple(out.size()), (4, 256, 7, 7))
        self.assertIs(x, out)

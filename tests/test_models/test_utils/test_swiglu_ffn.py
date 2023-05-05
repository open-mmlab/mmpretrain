# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn

from mmpretrain.models.utils import LayerScale, SwiGLUFFN, SwiGLUFFNFused


class TestSwiGLUFFN(TestCase):

    def test_init(self):
        swiglu = SwiGLUFFN(embed_dims=4)
        assert swiglu.w12.weight.shape == torch.ones((8, 4)).shape
        assert swiglu.w3.weight.shape == torch.ones((4, 4)).shape
        assert isinstance(swiglu.gamma2, nn.Identity)

        swiglu = SwiGLUFFN(embed_dims=4, layer_scale_init_value=0.1)
        assert isinstance(swiglu.gamma2, LayerScale)

    def test_forward(self):
        swiglu = SwiGLUFFN(embed_dims=4)
        x = torch.randn((1, 8, 4))
        out = swiglu(x)
        self.assertEqual(out.size(), x.size())

        swiglu = SwiGLUFFN(embed_dims=4, out_dims=12)
        x = torch.randn((1, 8, 4))
        out = swiglu(x)
        self.assertEqual(tuple(out.size()), (1, 8, 12))


class TestSwiGLUFFNFused(TestCase):

    def test_init(self):
        swiglu = SwiGLUFFNFused(embed_dims=4)
        assert swiglu.w12.weight.shape == torch.ones((16, 4)).shape
        assert swiglu.w3.weight.shape == torch.ones((4, 8)).shape
        assert isinstance(swiglu.gamma2, nn.Identity)

        swiglu = SwiGLUFFNFused(embed_dims=4, layer_scale_init_value=0.1)
        assert isinstance(swiglu.gamma2, LayerScale)

    def test_forward(self):
        swiglu = SwiGLUFFNFused(embed_dims=4)
        x = torch.randn((1, 8, 4))
        out = swiglu(x)
        self.assertEqual(out.size(), x.size())

        swiglu = SwiGLUFFNFused(embed_dims=4, out_dims=12)
        x = torch.randn((1, 8, 4))
        out = swiglu(x)
        self.assertEqual(tuple(out.size()), (1, 8, 12))

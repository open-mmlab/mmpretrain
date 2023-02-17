# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn.functional as F

from mmpretrain.models.utils import GRN, LayerNorm2d


class TestGRN(TestCase):

    def test_init(self):
        module = GRN(in_channels=32, eps=1e-3)
        self.assertEqual(module.in_channels, 32)
        self.assertEqual(module.eps, 1e-3)
        self.assertTrue(module.gamma.requires_grad)
        self.assertTrue(module.beta.requires_grad)
        self.assertEqual(module.gamma.shape, (32, ))
        self.assertTrue(module.beta.shape, (32, ))

    def test_forward(self):
        module = GRN(in_channels=32, eps=1e-3)
        input_ = torch.rand(1, 28, 28, 32)
        gx = torch.norm(input_, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=3, keepdim=True) + 1e-3)
        expected_out = module.gamma * input_ * nx + module.beta + input_

        torch.testing.assert_allclose(
            module(input_, data_format='channel_last'), expected_out)

        input_ = input_.permute([0, 3, 1, 2])
        expected_out = expected_out.permute([0, 3, 1, 2])
        torch.testing.assert_allclose(
            module(input_, data_format='channel_first'), expected_out)


class TestLayerNorm2d(TestCase):

    def test_init(self):
        module = LayerNorm2d(num_channels=32, eps=1e-3)
        self.assertEqual(module.num_channels, 32)
        self.assertEqual(module.eps, 1e-3)
        self.assertTrue(module.weight.requires_grad)
        self.assertTrue(module.bias.requires_grad)
        self.assertEqual(module.weight.shape, (32, ))
        self.assertTrue(module.bias.shape, (32, ))

    def test_forward(self):
        module = LayerNorm2d(num_channels=32, eps=1e-3)
        input_ = torch.rand(1, 28, 28, 32)
        expected_out = F.layer_norm(input_, module.normalized_shape,
                                    module.weight, module.bias, 1e-3)

        torch.testing.assert_allclose(
            module(input_, data_format='channel_last'), expected_out)

        input_ = input_.permute([0, 3, 1, 2])
        expected_out = expected_out.permute([0, 3, 1, 2])
        torch.testing.assert_allclose(
            module(input_, data_format='channel_first'), expected_out)

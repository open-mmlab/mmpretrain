# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import ANY, MagicMock

import pytest
import torch

from mmcls.models.utils.attention import ShiftWindowMSA, WindowMSA


def get_relative_position_index(window_size):
    """Method from original code of Swin-Transformer."""
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    # 2, Wh*Ww, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    # Wh*Ww, Wh*Ww, 2
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index


class TestWindowMSA(TestCase):

    def test_forward(self):
        attn = WindowMSA(embed_dims=96, window_size=(7, 7), num_heads=4)
        inputs = torch.rand((16, 7 * 7, 96))
        output = attn(inputs)
        self.assertEqual(output.shape, inputs.shape)

        # test non-square window_size
        attn = WindowMSA(embed_dims=96, window_size=(6, 7), num_heads=4)
        inputs = torch.rand((16, 6 * 7, 96))
        output = attn(inputs)
        self.assertEqual(output.shape, inputs.shape)

    def test_relative_pos_embed(self):
        attn = WindowMSA(embed_dims=96, window_size=(7, 8), num_heads=4)
        self.assertEqual(attn.relative_position_bias_table.shape,
                         ((2 * 7 - 1) * (2 * 8 - 1), 4))
        # test relative_position_index
        expected_rel_pos_index = get_relative_position_index((7, 8))
        self.assertTrue(
            torch.allclose(attn.relative_position_index,
                           expected_rel_pos_index))

        # test default init
        self.assertTrue(
            torch.allclose(attn.relative_position_bias_table,
                           torch.tensor(0.)))
        attn.init_weights()
        self.assertFalse(
            torch.allclose(attn.relative_position_bias_table,
                           torch.tensor(0.)))

    def test_qkv_bias(self):
        # test qkv_bias=True
        attn = WindowMSA(
            embed_dims=96, window_size=(7, 7), num_heads=4, qkv_bias=True)
        self.assertEqual(attn.qkv.bias.shape, (96 * 3, ))

        # test qkv_bias=False
        attn = WindowMSA(
            embed_dims=96, window_size=(7, 7), num_heads=4, qkv_bias=False)
        self.assertIsNone(attn.qkv.bias)

    def tets_qk_scale(self):
        # test default qk_scale
        attn = WindowMSA(
            embed_dims=96, window_size=(7, 7), num_heads=4, qk_scale=None)
        head_dims = 96 // 4
        self.assertAlmostEqual(attn.scale, head_dims**-0.5)

        # test specified qk_scale
        attn = WindowMSA(
            embed_dims=96, window_size=(7, 7), num_heads=4, qk_scale=0.3)
        self.assertEqual(attn.scale, 0.3)

    def test_attn_drop(self):
        inputs = torch.rand(16, 7 * 7, 96)
        attn = WindowMSA(
            embed_dims=96, window_size=(7, 7), num_heads=4, attn_drop=1.0)
        # drop all attn output, output shuold be equal to proj.bias
        self.assertTrue(torch.allclose(attn(inputs), attn.proj.bias))

    def test_prob_drop(self):
        inputs = torch.rand(16, 7 * 7, 96)
        attn = WindowMSA(
            embed_dims=96, window_size=(7, 7), num_heads=4, proj_drop=1.0)
        self.assertTrue(torch.allclose(attn(inputs), torch.tensor(0.)))

    def test_mask(self):
        inputs = torch.rand(16, 7 * 7, 96)
        attn = WindowMSA(embed_dims=96, window_size=(7, 7), num_heads=4)
        mask = torch.zeros((4, 49, 49))
        # Mask the first column
        mask[:, 0, :] = -100
        mask[:, :, 0] = -100
        outs = attn(inputs, mask=mask)
        inputs[:, 0, :].normal_()
        outs_with_mask = attn(inputs, mask=mask)
        torch.testing.assert_allclose(outs[:, 1:, :], outs_with_mask[:, 1:, :])


class TestShiftWindowMSA(TestCase):

    def test_forward(self):
        inputs = torch.rand((1, 14 * 14, 96))
        attn = ShiftWindowMSA(embed_dims=96, window_size=7, num_heads=4)
        output = attn(inputs, (14, 14))
        self.assertEqual(output.shape, inputs.shape)
        self.assertEqual(attn.w_msa.relative_position_bias_table.shape,
                         ((2 * 7 - 1)**2, 4))

        # test forward with shift_size
        attn = ShiftWindowMSA(
            embed_dims=96, window_size=7, num_heads=4, shift_size=3)
        output = attn(inputs, (14, 14))
        assert output.shape == (inputs.shape)

        # test irregular input shape
        input_resolution = (19, 18)
        attn = ShiftWindowMSA(embed_dims=96, num_heads=4, window_size=7)
        inputs = torch.rand((1, 19 * 18, 96))
        output = attn(inputs, input_resolution)
        assert output.shape == (inputs.shape)

        # test wrong input_resolution
        input_resolution = (14, 14)
        attn = ShiftWindowMSA(embed_dims=96, num_heads=4, window_size=7)
        inputs = torch.rand((1, 14 * 14, 96))
        with pytest.raises(AssertionError):
            attn(inputs, (14, 15))

    def test_pad_small_map(self):
        # test pad_small_map=True
        inputs = torch.rand((1, 6 * 7, 96))
        attn = ShiftWindowMSA(
            embed_dims=96,
            window_size=7,
            num_heads=4,
            shift_size=3,
            pad_small_map=True)
        attn.get_attn_mask = MagicMock(wraps=attn.get_attn_mask)
        output = attn(inputs, (6, 7))
        self.assertEqual(output.shape, inputs.shape)
        attn.get_attn_mask.assert_called_once_with((7, 7),
                                                   window_size=7,
                                                   shift_size=3,
                                                   device=ANY)

        # test pad_small_map=False
        inputs = torch.rand((1, 6 * 7, 96))
        attn = ShiftWindowMSA(
            embed_dims=96,
            window_size=7,
            num_heads=4,
            shift_size=3,
            pad_small_map=False)
        with self.assertRaisesRegex(AssertionError, r'the window size \(7\)'):
            attn(inputs, (6, 7))

        # test pad_small_map=False, and the input size equals to window size
        inputs = torch.rand((1, 7 * 7, 96))
        attn.get_attn_mask = MagicMock(wraps=attn.get_attn_mask)
        output = attn(inputs, (7, 7))
        self.assertEqual(output.shape, inputs.shape)
        attn.get_attn_mask.assert_called_once_with((7, 7),
                                                   window_size=7,
                                                   shift_size=0,
                                                   device=ANY)

    def test_drop_layer(self):
        inputs = torch.rand((1, 14 * 14, 96))
        attn = ShiftWindowMSA(
            embed_dims=96,
            window_size=7,
            num_heads=4,
            dropout_layer=dict(type='Dropout', drop_prob=1.0))
        attn.init_weights()
        # drop all attn output, output shuold be equal to proj.bias
        self.assertTrue(
            torch.allclose(attn(inputs, (14, 14)), torch.tensor(0.)))

    def test_deprecation(self):
        # test deprecated arguments
        with pytest.warns(DeprecationWarning):
            ShiftWindowMSA(
                embed_dims=96,
                num_heads=4,
                window_size=7,
                input_resolution=(14, 14))

        with pytest.warns(DeprecationWarning):
            ShiftWindowMSA(
                embed_dims=96, num_heads=4, window_size=7, auto_pad=True)

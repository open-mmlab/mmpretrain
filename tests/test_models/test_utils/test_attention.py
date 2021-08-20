# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
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


def test_window_msa():
    batch_size = 1
    num_windows = (4, 4)
    embed_dims = 96
    window_size = (7, 7)
    num_heads = 4
    attn = WindowMSA(
        embed_dims=embed_dims, window_size=window_size, num_heads=num_heads)
    inputs = torch.rand((batch_size * num_windows[0] * num_windows[1],
                         window_size[0] * window_size[1], embed_dims))

    # test forward
    output = attn(inputs)
    assert output.shape == inputs.shape
    assert attn.relative_position_bias_table.shape == (
        (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)

    # test relative_position_bias_table init
    attn.init_weights()
    assert abs(attn.relative_position_bias_table).sum() > 0

    # test non-square window_size
    window_size = (6, 7)
    attn = WindowMSA(
        embed_dims=embed_dims, window_size=window_size, num_heads=num_heads)
    inputs = torch.rand((batch_size * num_windows[0] * num_windows[1],
                         window_size[0] * window_size[1], embed_dims))
    output = attn(inputs)
    assert output.shape == inputs.shape

    # test relative_position_index
    expected_rel_pos_index = get_relative_position_index(window_size)
    assert (attn.relative_position_index == expected_rel_pos_index).all()

    # test qkv_bias=True
    attn = WindowMSA(
        embed_dims=embed_dims,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=True)
    assert attn.qkv.bias.shape == (embed_dims * 3, )

    # test qkv_bias=False
    attn = WindowMSA(
        embed_dims=embed_dims,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=False)
    assert attn.qkv.bias is None

    # test default qk_scale
    attn = WindowMSA(
        embed_dims=embed_dims,
        window_size=window_size,
        num_heads=num_heads,
        qk_scale=None)
    head_dims = embed_dims // num_heads
    assert np.isclose(attn.scale, head_dims**-0.5)

    # test specified qk_scale
    attn = WindowMSA(
        embed_dims=embed_dims,
        window_size=window_size,
        num_heads=num_heads,
        qk_scale=0.3)
    assert attn.scale == 0.3

    # test attn_drop
    attn = WindowMSA(
        embed_dims=embed_dims,
        window_size=window_size,
        num_heads=num_heads,
        attn_drop=1.0)
    inputs = torch.rand((batch_size * num_windows[0] * num_windows[1],
                         window_size[0] * window_size[1], embed_dims))
    # drop all attn output, output shuold be equal to proj.bias
    assert torch.allclose(attn(inputs), attn.proj.bias)

    # test prob_drop
    attn = WindowMSA(
        embed_dims=embed_dims,
        window_size=window_size,
        num_heads=num_heads,
        proj_drop=1.0)
    assert (attn(inputs) == 0).all()


def test_shift_window_msa():
    batch_size = 1
    embed_dims = 96
    input_resolution = (14, 14)
    num_heads = 4
    window_size = 7

    # test forward
    attn = ShiftWindowMSA(
        embed_dims=embed_dims,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size)
    inputs = torch.rand(
        (batch_size, input_resolution[0] * input_resolution[1], embed_dims))
    output = attn(inputs)
    assert output.shape == (inputs.shape)
    assert attn.w_msa.relative_position_bias_table.shape == ((2 * window_size -
                                                              1)**2, num_heads)

    # test forward with shift_size
    attn = ShiftWindowMSA(
        embed_dims=embed_dims,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=1)
    output = attn(inputs)
    assert output.shape == (inputs.shape)

    # test relative_position_bias_table init
    attn.init_weights()
    assert abs(attn.w_msa.relative_position_bias_table).sum() > 0

    # test dropout_layer
    attn = ShiftWindowMSA(
        embed_dims=embed_dims,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        dropout_layer=dict(type='DropPath', drop_prob=0.5))
    torch.manual_seed(0)
    output = attn(inputs)
    assert (output == 0).all()

    # test auto_pad
    input_resolution = (19, 18)
    attn = ShiftWindowMSA(
        embed_dims=embed_dims,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        auto_pad=True)
    assert attn.pad_r == 3
    assert attn.pad_b == 2

    # test small input_resolution
    input_resolution = (5, 6)
    attn = ShiftWindowMSA(
        embed_dims=embed_dims,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=3,
        auto_pad=True)
    assert attn.window_size == 5
    assert attn.shift_size == 0

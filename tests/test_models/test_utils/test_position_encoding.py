# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpretrain.models.utils import (ConditionalPositionEncoding,
                                     RotaryEmbeddingFast)


def test_conditional_position_encoding_module():
    CPE = ConditionalPositionEncoding(in_channels=32, embed_dims=32, stride=2)
    outs = CPE(torch.randn(1, 3136, 32), (56, 56))
    assert outs.shape == torch.Size([1, 784, 32])


def test_rotary_embedding_fast_module():
    RoPE = RotaryEmbeddingFast(embed_dims=64, patch_resolution=24)
    outs = RoPE(torch.randn(1, 2, 24 * 24, 64), (24, 24))
    assert outs.shape == torch.Size([1, 2, 24 * 24, 64])

    RoPE = RotaryEmbeddingFast(embed_dims=64, patch_resolution=(14, 20))
    outs = RoPE(torch.randn(1, 2, 14 * 20, 64), (14, 20))
    assert outs.shape == torch.Size([1, 2, 14 * 20, 64])

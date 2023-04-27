# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_scale import LayerScale


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN layer.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """  # noqa

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.out_dims = out_dims or embed_dims
        hidden_dims = feedforward_channels or embed_dims

        self.w12 = nn.Linear(self.embed_dims, 2 * hidden_dims, bias=bias)
        self.w3 = nn.Linear(hidden_dims, self.out_dims, bias=bias)

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(
                dim=embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        out = self.w3(hidden)
        out = self.gamma2(out)
        if self.out_dims != self.embed_dims:
            # due to the dimension inconsistence, noto apply residual operation
            return out

        if identity is None:
            identity = x
        return identity + out


class SwiGLUFFNFused(SwiGLUFFN):
    """SwiGLU FFN layer with fusing.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """  # noqa

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.,
        bias: bool = True,
    ) -> None:
        out_dims = out_dims or embed_dims
        feedforward_channels = feedforward_channels or embed_dims
        feedforward_channels = (int(feedforward_channels * 2 / 3) + 7) // 8 * 8
        super().__init__(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            out_dims=out_dims,
            layer_scale_init_value=layer_scale_init_value,
            bias=bias,
        )

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule
from torch import nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class DINONeck(BaseModule):
    """Implementation for DINO neck.

    This module is proposed in `DINO: Emerging Properties in Self-Supervised
    Vision Transformers <https://arxiv.org/abs/2104.14294>`_.

    Args:
        in_channels (int): Input channels.
        hidden_channels (int): Hidden channels.
        out_channels (int): Output channels.
        bottleneck_channels (int): Bottleneck channels.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, bottleneck_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(*[
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, bottleneck_channels),
        ])

        self.last_layer = nn.Linear(
            bottleneck_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x[0])
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

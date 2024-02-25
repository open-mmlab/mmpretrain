# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class SimMIMLinearDecoder(BaseModule):
    """Linear Decoder For SimMIM pretraining.

    This neck reconstructs the original image from the shrunk feature map.

    Args:
        in_channels (int): Channel dimension of the feature map.
        encoder_stride (int): The total stride of the encoder.
        target_channels (int): Channel dimensions of original image.
    """

    def __init__(self, in_channels: int, encoder_stride: int, target_channels: int = 3) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=encoder_stride**2 * target_channels,
                kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.decoder(x)
        return x

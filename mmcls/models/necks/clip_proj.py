# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmcls.registry import MODELS


@MODELS.register_module()
class CLIPProjection(BaseModule):
    """Neck with CLIP Projection.

    You can check in [clip repo](https://github.com/mlfoundations/open_clip/
    blob/v2.13.0/src/open_clip/transformer.py#L391).
    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 init_cfg: Optional[dict] = None):
        super(CLIPProjection, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        scale = in_channels**-0.5
        self.proj = nn.Parameter(scale *
                                 torch.randn(in_channels, out_channels))

    def forward(self, inputs: Tuple) -> Tuple[torch.Tensor]:
        """forward function.

        Args:
            inputs (Tuple): The features extracted from
                 the backbone. Multiple stage inputs are acceptable but only
                  the last stage will be used.
        Returns:
            Tuple(torch.Tensor)): A tuple of reducted features.
        """
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
            out = inputs @ self.proj
        elif isinstance(inputs, torch.Tensor):
            out = inputs @ self.proj
        else:
            raise TypeError(
                '`CLIPProjection` neck inputs should be tuple or torch.tensor')
        return (out, )

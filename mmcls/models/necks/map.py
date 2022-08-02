# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class MultiheadAttentionPooling(BaseModule):
    """MultiheadAttentionPooling.

    Args:
        in_channels (list[int]): The input channels of all scales.
        num_heads (list[ing]): The number of heads of all pooling map.
            Default to 8.
        out_channels (int): The channels of used feature map.
            Default to 2048.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01))``.
    """

    def __init__(self,
                 in_channels,
                 num_heads,
                 out_channels=2048,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(MultiheadAttentionPooling, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.norm_cfg = norm_cfg

        assert len(self.in_channels) == len(
            self.num_heads
        ), 'The in_channels and num_heads must have the same length'

        mhsa = list()
        for i in range(len(in_channels)):
            mhsa.append(
                MultiheadAttention(
                    embed_dims=in_channels[i],
                    num_heads=num_heads[i],
                    batch_first=True,
                ), )

        self.mhsa = nn.ModuleList(mhsa)

        final_layers = list()
        for i in range(len(in_channels)):
            final_layer = nn.Sequential(
                ConvModule(
                    in_channels=in_channels[i],
                    out_channels=out_channels,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    bias=False,
                ),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            final_layers.append(final_layer)

        self.final_layers = nn.ModuleList(final_layers)

    def mhsa_forward(self, x, model):
        B, nc, h, w = x.shape
        x = x.flatten(-2, -1).permute(0, 2, 1)
        x = model(x)
        x = x.permute(0, 2, 1)
        x = x.view(B, nc, h, w)
        return x.contiguous()

    def forward(self, x):
        assert isinstance(x, tuple) and len(x) == len(self.in_channels)

        feats = [
            self.mhsa_forward(x[i], self.mhsa[i])
            for i in range(len(self.in_channels))
        ]
        feats = [
            self.final_layers[i](feats[i])
            for i in range(len(self.in_channels))
        ]

        feats = torch.stack(feats, dim=0).sum(dim=0)
        return feats

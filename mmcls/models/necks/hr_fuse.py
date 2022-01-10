# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import BaseModule

from ..backbones.resnet import Bottleneck, ResLayer
from ..builder import NECKS


@NECKS.register_module()
class HRFuseScales(BaseModule):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(HRFuseScales, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.norm_cfg = norm_cfg

        block_type = Bottleneck
        out_channels = [128, 256, 512, 1024]

        # Increase the channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        increase_layers = []
        for i in range(len(in_channels)):
            increase_layers.append(
                ResLayer(
                    block_type,
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    num_blocks=1,
                    stride=1,
                ))
        self.increase_layers = nn.ModuleList(increase_layers)

        # Downsample feature maps in each scale.
        downsample_layers = []
        for i in range(len(in_channels) - 1):
            downsample_layers.append(
                ConvModule(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=True,
                ))
        self.downsample_layers = nn.ModuleList(downsample_layers)

        # The final conv block before final classifier linear layer.
        self.final_layer = ConvModule(
            in_channels=out_channels[3],
            out_channels=2048,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            bias=True,
        )

    def forward(self, x):
        assert isinstance(x, tuple) and len(x) == len(self.in_channels)

        feat = self.increase_layers[0](x[0])
        for i in range(len(self.downsample_layers)):
            feat = self.downsample_layers[i](feat)
            feat += self.increase_layers[i + 1](x[i + 1])

        return self.final_layer(feat)

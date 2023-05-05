# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class MobileNetV1(BaseBackbone):

    def __init__(self,
                 input_channels,
                 conv_cfg=None,
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(MobileNetV1, self).__init__(init_cfg)
        self.arch_settings = [[32, 64, 1], [64, 128, 2], [128, 128, 1],
                              [128, 256, 2], [256, 256, 1], [256, 512, 2],
                              [512, 512, 1], [512, 512, 1], [512, 512, 1],
                              [512, 512, 1], [512, 512, 1], [512, 1024, 2],
                              [1024, 1024, 1]]
        self.in_channels = input_channels
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.layers = []
        layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.layers.append(layer)
        for layer_cfg in (self.arch_settings):
            in_ch, out_ch, stride = layer_cfg
            self.layers.append(
                ConvModule(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(0, self.frozen_stages + 1):
                layer = getattr(self, f'layer{i}')
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(MobileNetV1, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

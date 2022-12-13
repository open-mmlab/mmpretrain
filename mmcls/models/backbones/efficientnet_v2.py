# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Callable, List, Optional, Tuple, Sequence

import torch
from debugpy.common.log import warning
from torch import Tensor
import torch.nn as nn
from mmcv.cnn.bricks import DropPath, ConvModule
from mmengine.model import BaseModule, Sequential

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.backbones.efficientnet import EdgeResidual as FusedMBConv
from mmcls.models.utils import InvertedResidual as MBConv
from mmcls.registry import MODELS


class EnhancedConvModule(ConvModule):
    """Conv with short-cut and droppath."""
    def __init__(self,*args, has_skip=False, drop_path_rate=0, **args):
        super()._init__(*args, **args)
        # some assert lines
        self.has_skip = has_skip
        if self.has_skip:
            assert stride == 1 and in_channels == out_channels, (
                "the stride must be 1 and the in_C must be equal with"
                " out_C, when `skip` is True in `ConvWithSkip` .")
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
        if self.has_skip and self.inplace :
            self.inplace = False
            warning.warn("if there has short-cut in `EnhanceConv`, in place would be False.")

    def forward(self,
                x: torch.Tensor,
                activate: bool = True,
                norm: bool = True) -> torch.Tensor:
        short_cut = x
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        if self.has_skip:
            x = self.drop_path(x) + short_cut
        return x


# class ConvWithSkip(BaseModule):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  skip=True,
#                  drop_path_rate=0.,
#                  conv_cfg=dict(type='Conv2dAdaptivePadding'),
#                  norm_cfg=dict(type='BN', eps=1e-3, momentum=0.1),
#                  act_cfg=dict(type='Swish'),
#                  init_cfg=None):
#         super(ConvWithSkip, self).__init__(init_cfg=init_cfg)
#         self.conv = ConvModule(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg)
#         self.has_skip = skip and stride == 1 and in_channels == out_channels
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
#
#     def forward(self, x):
#         shortcut = x
#         x = self.conv(x)
#         if self.has_skip:
#             x = self.drop_path(x) + shortcut
#         return x


@MODELS.register_module()
class EfficientNetV2(BaseBackbone):
    # repeat, kernel, stride, expansion, in_c, out_c, se_ratio, block_type
    # if block_type==0, is FusedMBConv; if block_type==1, is MBConv,
    # if block_type==-1, is ConvWithSkip; if block_type==-2, is ConvModule
    # b0 is same as base
    arch_settings = {
        'base': [[1, 3, 1, 1, 32, 16, 0, -1], [2, 3, 2, 4, 16, 32, 0, 0],
                 [2, 3, 2, 4, 32, 48, 0, 0], [3, 3, 2, 4, 48, 96, 0.25, 1],
                 [5, 3, 1, 6, 96, 112, 0.25, 1],
                 [8, 3, 2, 6, 112, 192, 0.25, 1],
                 [1, 1, 1, 1, 192, 1280, 0, -2]],
        's': [[2, 3, 1, 1, 24, 24, 0, -1], [4, 3, 2, 4, 24, 48, 0, 0],
              [4, 3, 2, 4, 48, 64, 0, 0], [6, 3, 2, 4, 64, 128, 0.25, 1],
              [9, 3, 1, 6, 128, 160, 0.25, 1],
              [15, 3, 2, 6, 160, 256, 0.25, 1],
              [1, 1, 1, 1, 256, 1280, 0, -2]],
        'm': [[3, 3, 1, 1, 24, 24, 0, -1], [5, 3, 2, 4, 24, 48, 0, 0],
              [5, 3, 2, 4, 48, 80, 0, 0], [7, 3, 2, 4, 80, 160, 0.25, 1],
              [14, 3, 1, 6, 160, 176, 0.25, 1],
              [18, 3, 2, 6, 176, 304, 0.25, 1],
              [5, 3, 1, 6, 304, 512, 0.25, 1],
              [1, 1, 1, 1, 512, 1280, 0, -2]],
        'l': [[4, 3, 1, 1, 32, 32, 0, -1], [7, 3, 2, 4, 32, 64, 0, 0],
              [7, 3, 2, 4, 64, 96, 0, 0], [10, 3, 2, 4, 96, 192, 0.25, 1],
              [19, 3, 1, 6, 192, 224, 0.25, 1],
              [25, 3, 2, 6, 224, 384, 0.25, 1],
              [7, 3, 1, 6, 384, 640, 0.25, 1],
              [1, 1, 1, 1, 640, 1280, 0, -2]],
        'xl': [[4, 3, 1, 1, 32, 32, 0, -1], [8, 3, 2, 4, 32, 64, 0, 0],
               [8, 3, 2, 4, 64, 96, 0, 0], [16, 3, 2, 4, 96, 192, 0.25, 1],
               [24, 3, 1, 6, 192, 256, 0.25, 1],
               [32, 3, 2, 6, 256, 512, 0.25, 1],
               [8, 3, 1, 6, 512, 640, 0.25, 1],
               [1, 1, 1, 1, 640, 1280, 0, -2]],
        'b0': [[1, 3, 1, 1, 32, 16, 0, -1], [2, 3, 2, 4, 16, 32, 0, 0],
               [2, 3, 2, 4, 32, 48, 0, 0], [3, 3, 2, 4, 48, 96, 0.25, 1],
               [5, 3, 1, 6, 96, 112, 0.25, 1],
               [8, 3, 2, 6, 112, 192, 0.25, 1],
               [1, 1, 1, 1, 192, 1280, 0, -2]],
        'b1': [[2, 3, 1, 1, 32, 16, 0, -1], [3, 3, 2, 4, 16, 32, 0, 0],
               [3, 3, 2, 4, 32, 48, 0, 0], [4, 3, 2, 4, 48, 96, 0.25, 1],
               [6, 3, 1, 6, 96, 112, 0.25, 1],
               [9, 3, 2, 6, 112, 192, 0.25, 1],
               [1, 1, 1, 1, 192, 1280, 0, -2]],
        'b2': [[2, 3, 1, 1, 32, 16, 0, -1], [3, 3, 2, 4, 16, 32, 0, 0],
               [3, 3, 2, 4, 32, 56, 0, 0], [4, 3, 2, 4, 56, 104, 0.25, 1],
               [6, 3, 1, 6, 104, 120, 0.25, 1],
               [10, 3, 2, 6, 120, 208, 0.25, 1],
               [1, 1, 1, 1, 208, 1408, 0, -2]],
        'b3': [[2, 3, 1, 1, 40, 16, 0, -1], [3, 3, 2, 4, 16, 40, 0, 0],
               [3, 3, 2, 4, 40, 56, 0, 0], [5, 3, 2, 4, 56, 112, 0.25, 1],
               [7, 3, 1, 6, 112, 136, 0.25, 1],
               [12, 3, 2, 6, 136, 232, 0.25, 1],
               [1, 1, 1, 1, 232, 1536, 0, -2]]
    }

    def __init__(self,
                 arch: str = 's',
                 drop_path_rate: float = 0.,
                 out_indices: Tuple = (-1,),
                 frozen_stages: int = 0,
                 conv_cfg=dict(type='Conv2dAdaptivePadding'),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.1),
                 act_cfg=dict(type='Swish'),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(
                         type='Constant',
                         layer=['_BatchNorm', 'GroupNorm'],
                         val=1)
                 ]):
        super(EfficientNetV2, self).__init__(init_cfg)
        assert arch in self.arch_settings, \
            f'"{arch}" is not one of the arch_settings ' \
            f'({", ".join(self.arch_settings.keys())})'
        self.arch = self.arch_settings[arch]
        self.drop_path_rate = drop_path_rate
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.layers = nn.ModuleList()
        block_cfg_last = self.arch[-1]
        assert block_cfg_last[-1] == -2, \
            f'the last block_type must be -2 ,'\
            f'It is conv module. but ' \
            f'get block_type={block_cfg_last[-1]}'
        self.in_channels = self.arch[0][4]
        self.out_channels = block_cfg_last[5]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.layers.append(
            ConvModule(
                in_channels=3,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))  # the first conv not in arch
        self.make_layer()
        self.layers.append(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=block_cfg_last[1],
                stride=block_cfg_last[2],
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = len(self.layers) + index
            assert 0 <= out_indices[i] <= len(self.layers), \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

    def make_layer(self):
        layer_setting = self.arch[:-1]

        total_num_blocks = sum([x[0] for x in layer_setting])
        block_idx = 0
        dpr = [
            x.item()
            for x in torch.linspace(0, self.drop_path_rate, total_num_blocks)
        ]  # stochastic depth decay rule

        for layer_cfg in layer_setting:
            layer = []
            (repeat, kernel_size, stride, expand_ratio, _, out_channels,
             se_ratio, block_type) = layer_cfg
            for i in range(repeat):
                stride = stride if i == 0 else 1
                if block_type == -1:
                    if stride == 1 and self.in_channels == out_channels:
                        layer.append(EnhancedConvModule
                            (in_channels=self.in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             has_skip=True,
                             drop_path_rate=dpr[block_idx],
                             conv_cfg=self.conv_cfg,
                             norm_cfg=self.norm_cfg,
                             act_cfg=self.act_cfg))
                        self.in_channels = out_channels
                    else:
                        layer.append(ConvModule(
                            in_channels=self.in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                        self.in_channels = out_channels
                else:
                    mid_channels = int(self.in_channels * expand_ratio)
                    if se_ratio <= 0:
                        se_cfg = None
                    else:
                        se_cfg = dict(
                            channels=mid_channels,
                            ratio=expand_ratio * (1.0/se_ratio),
                            divisor=1,
                            act_cfg=(self.act_cfg, dict(type='Sigmoid')))
                    if block_type == 0:
                        se_cfg = None
                        block = FusedMBConv
                    else:
                        block = MBConv
                    layer.append(
                        block(
                            in_channels=self.in_channels,
                            out_channels=out_channels,
                            mid_channels=mid_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            se_cfg=se_cfg,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            drop_path_rate=dpr[block_idx],
                            with_cp=self.with_cp))
                    self.in_channels = out_channels
                block_idx += 1
            self.layers.append(Sequential(*layer))

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(EfficientNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

# modified from pycls
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""EfficientNet models."""

import warnings
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks import ACTIVATION_LAYERS, build_activation_layer
from mmcv.runner.base_module import BaseModule

from ..builder import BACKBONES


class SiLU(BaseModule):
    """SiLU activation function (also known as Swish): x * sigmoid(x)."""

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


if hasattr(nn, 'SiLU'):
    ACTIVATION_LAYERS.register_module(module=nn.SiLU)
else:
    ACTIVATION_LAYERS.register_module(module=SiLU)


def conv2d(in_channels,
           out_channels,
           kernel_size,
           *,
           stride=1,
           groups=1,
           bias=False):
    """Helper for building a conv2d layer."""
    assert kernel_size % 2 == 1, 'Only odd size kernels supported ' \
                                 'to avoid padding issues.'
    pad = (kernel_size - 1) // 2
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=pad,
        groups=groups,
        bias=bias)


def norm2d(in_channels):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=in_channels, eps=1e-5, momentum=0.1)


def linear(in_channels, out_channels, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Linear(in_channels, out_channels, bias=bias)


class SE(BaseModule):
    """Squeeze-and-Excitation (SE) block."""

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            conv2d(in_channels, se_channels, 1, bias=True),
            build_activation_layer(dict(type='SiLU')),
            conv2d(se_channels, in_channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class EffHead(BaseModule):
    """EfficientNet head."""

    def __init__(self, in_channels, out_channels):
        super(EffHead, self).__init__()
        self.conv = conv2d(in_channels, out_channels, 1)
        self.conv_bn = norm2d(out_channels)
        self.conv_af = build_activation_layer(dict(type='SiLU'))

    def forward(self, x):
        x = self.conv_af(self.conv_bn(self.conv(x)))
        return x


class MBConv(BaseModule):
    """Mobile inverted bottleneck block with SE."""

    def __init__(self, in_channels, expansion_ratio, kernel_size, stride,
                 se_ratio, out_channel):
        super(MBConv, self).__init__()
        self.exp = None
        expansion_channels = int(in_channels * expansion_ratio)
        if expansion_channels != in_channels:
            self.exp = conv2d(in_channels, expansion_channels, 1)
            self.exp_bn = norm2d(expansion_channels)
            self.exp_af = build_activation_layer(dict(type='SiLU'))
        self.dwise = conv2d(
            expansion_channels,
            expansion_channels,
            kernel_size,
            stride=stride,
            groups=expansion_channels)
        self.dwise_bn = norm2d(expansion_channels)
        self.dwise_af = build_activation_layer(dict(type='SiLU'))
        self.se = SE(expansion_channels, int(in_channels * se_ratio))
        self.lin_proj = conv2d(expansion_channels, out_channel, 1)
        self.lin_proj_bn = norm2d(out_channel)
        self.has_skip = stride == 1 and in_channels == out_channel

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            f_x = x + f_x
        return f_x


class EffStage(BaseModule):
    """EfficientNet stage."""

    def __init__(self, in_channels, expansion_ratio, kernel_size, stride,
                 se_ratio, out_channels, depth):
        super(EffStage, self).__init__()
        for i in range(depth):
            block = MBConv(in_channels, expansion_ratio, kernel_size, stride,
                           se_ratio, out_channels)
            self.add_module('b{}'.format(i + 1), block)
            stride, in_channels = 1, out_channels

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


@BACKBONES.register_module()
class EfficientNet(BaseModule):
    """EfficientNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1905.11946>`__ for
    details.

    Args:
        arch (ste): Network architecture, from {b0, b1, b2, b3, b4, b5}.
            Defaults to None.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_width (int): Output channels of the stem layer. Defaults to None.
        depths (Sequence[int]): Depth of each stage. Defaults to None.
        widths (Sequence[int]): Channels of each stage. Defaults to None.
        expansion_ratios (Sequence[int]): Expansion ratio of each stage.
            Defaults to None.
        strides (Sequence[int]): Stride of  each stage. Defaults to None.
        kernel_sizes (Sequence[int]): Kernel size of each stage. Defaults
            to None.
        head_width (int): Channels of the output feature map. Defaults to None.
        se_ratio (int): The ratio of the Squeeze-Excitation module.  Defaults
            to 0.25.
        out_indices (Sequence | int): Output from which layer. Defaults to -1,
            means the last layer.
    """

    arch_zoo = dict(
        b0=dict(
            stem_width=32,
            depths=[1, 2, 2, 3, 3, 4, 1],
            widths=[16, 24, 40, 80, 112, 192, 320],
            expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            strides=[1, 2, 2, 2, 1, 2, 1],
            kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            head_width=1280),
        b1=dict(
            stem_width=32,
            depths=[2, 3, 3, 4, 4, 5, 2],
            widths=[16, 24, 40, 80, 112, 192, 320],
            expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            strides=[1, 2, 2, 2, 1, 2, 1],
            kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            head_width=1280),
        b2=dict(
            stem_width=32,
            depths=[2, 3, 3, 4, 4, 5, 2],
            widths=[16, 24, 48, 88, 120, 208, 352],
            expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            strides=[1, 2, 2, 2, 1, 2, 1],
            kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            head_width=1408),
        b3=dict(
            stem_width=40,
            depths=[2, 3, 3, 5, 5, 6, 2],
            widths=[24, 32, 48, 96, 136, 232, 384],
            expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            strides=[1, 2, 2, 2, 1, 2, 1],
            kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            head_width=1536),
        b4=dict(
            stem_width=48,
            depths=[2, 4, 4, 6, 6, 8, 2],
            widths=[24, 32, 56, 112, 160, 272, 448],
            expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            strides=[1, 2, 2, 2, 1, 2, 1],
            kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            head_width=1792),
        b5=dict(
            stem_width=48,
            depths=[3, 5, 5, 7, 7, 9, 3],
            widths=[24, 40, 64, 128, 176, 304, 512],
            expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            strides=[1, 2, 2, 2, 1, 2, 1],
            kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            head_width=2048),
    )

    def __init__(self,
                 arch=None,
                 stem_width=None,
                 depths=None,
                 widths=None,
                 expansion_ratios=None,
                 strides=None,
                 kernel_sizes=None,
                 head_width=None,
                 in_channels=3,
                 se_ratio=0.25,
                 out_indices=-1):
        super(EfficientNet, self).__init__()

        if arch is not None:
            assert isinstance(arch, str), 'Unknown arch'
            for param in [
                    stem_width, depths, widths, expansion_ratios, strides,
                    kernel_sizes, head_width
            ]:
                if param is not None:
                    warnings.warn('specifying arch will overwrite the other '
                                  'parameters')
            arch = arch.lower()
            assert arch in EfficientNet.arch_zoo.keys()
            arch_setting = EfficientNet.arch_zoo[arch]
            stem_width = arch_setting['stem_width']
            depths = arch_setting['depths']
            widths = arch_setting['widths']
            expansion_ratios = arch_setting['expansion_ratios']
            strides = arch_setting['strides']
            kernel_sizes = arch_setting['kernel_sizes']
            head_width = arch_setting['head_width']

        stage_params = list(
            zip(depths, widths, expansion_ratios, strides, kernel_sizes))
        self.stem = nn.Sequential(
            conv2d(in_channels, stem_width, 3, stride=2), norm2d(stem_width),
            build_activation_layer(dict(type='SiLU')))
        prev_width = stem_width
        for i, (depth, width, exp_ratio, stride,
                kernel) in enumerate(stage_params):
            stage = EffStage(prev_width, exp_ratio, kernel, stride, se_ratio,
                             width, depth)
            self.add_module('s{}'.format(i + 1), stage)
            prev_width = width
        self.head = EffHead(prev_width, head_width)

        num_layers = len(depths) + 2  # stem + head
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must be a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = num_layers + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
            else:
                assert index >= self.num_layers, f'Invalid out_indices {index}'
        self.out_indices = out_indices

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                zero_init_gamma = False
                zero_init_gamma = hasattr(
                    m, 'final_bn') and m.final_bn and zero_init_gamma
                m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, x):
        result = []
        for idx, module in enumerate(self.children()):
            x = module(x)
            if idx in self.out_indices:
                result.append(x)
        return tuple(result)

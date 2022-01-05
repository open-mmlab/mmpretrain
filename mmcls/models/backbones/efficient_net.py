# modified from pycls
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""EfficientNet models."""

import warnings

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


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, 'Only odd size kernels supported ' \
                       'to avoid padding issues.'
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)


def norm2d(w_in):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=1e-5, momentum=0.1)


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Linear(w_in, w_out, bias=bias)


class SE(BaseModule):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_se, 1, bias=True),
            build_activation_layer(dict(type='SiLU')),
            conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class EffHead(BaseModule):
    """EfficientNet head: 1x1, BN, AF, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out):
        super(EffHead, self).__init__()
        self.conv = conv2d(w_in, w_out, 1)
        self.conv_bn = norm2d(w_out)
        self.conv_af = build_activation_layer(dict(type='SiLU'))

    def forward(self, x):
        x = self.conv_af(self.conv_bn(self.conv(x)))
        return x


class MBConv(BaseModule):
    """Mobile inverted bottleneck block with SE."""

    def __init__(self, in_channel, expansion_ratio, kernel_size, stride,
                 se_ratio, out_channel):
        # Expansion, kxk dwise, BN, AF, SE, 1x1, BN, skip_connection
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(in_channel * expansion_ratio)
        if w_exp != in_channel:
            self.exp = conv2d(in_channel, w_exp, 1)
            self.exp_bn = norm2d(w_exp)
            self.exp_af = build_activation_layer(dict(type='SiLU'))
        self.dwise = conv2d(
            w_exp, w_exp, kernel_size, stride=stride, groups=w_exp)
        self.dwise_bn = norm2d(w_exp)
        self.dwise_af = build_activation_layer(dict(type='SiLU'))
        self.se = SE(w_exp, int(in_channel * se_ratio))
        self.lin_proj = conv2d(w_exp, out_channel, 1)
        self.lin_proj_bn = norm2d(out_channel)
        self.has_skip = stride == 1 and in_channel == out_channel

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
class EffNet(BaseModule):
    """EfficientNet model."""

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
                 se_ratio=0.25):
        super(EffNet, self).__init__()

        if arch is not None:
            assert isinstance(arch, str)
            for param in [
                    stem_width, depths, widths, expansion_ratios, strides,
                    kernel_sizes, head_width
            ]:
                if param is not None:
                    warnings.warn('specifying arch will overwrite the other '
                                  'parameters')
            arch = arch.lower()
            assert arch in EffNet.arch_zoo.keys()
            arch_setting = EffNet.arch_zoo[arch]
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
            conv2d(3, stem_width, 3, stride=2), norm2d(stem_width),
            build_activation_layer(dict(type='SiLU')))
        prev_width = stem_width
        for i, (depth, width, exp_ratio, stride,
                kernel) in enumerate(stage_params):
            stage = EffStage(prev_width, exp_ratio, kernel, stride, se_ratio,
                             width, depth)
            self.add_module('s{}'.format(i + 1), stage)
            prev_width = width
        self.head = EffHead(prev_width, head_width)

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
        for module in self.children():
            x = module(x)
        return (x, )

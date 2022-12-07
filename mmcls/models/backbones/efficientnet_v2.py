# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn.bricks import DropPath
from mmengine.model import Sequential
from torch import Tensor

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.registry import MODELS


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ConvBNAct(nn.Module):

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 padding: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish

        if stride == 1:
            self.conv = nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False)
        elif stride == 2:
            self.conv = Conv2dSame(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer(inplace=True)

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result


class SqueezeExcite(nn.Module):

    def __init__(
            self,
            input_c: int,  # block input channel
            expand_c: int,  # block expand channel
            se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU(inplace=True)  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class MBConv(nn.Module):

    def __init__(self, kernel_size: int, input_c: int, out_c: int,
                 expand_ratio: int, stride: int, se_ratio: float,
                 drop_rate: float, norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError('illegal stride value.')

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(
            input_c,
            expanded_c,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(
            expanded_c,
            expanded_c,
            kernel_size=kernel_size,
            stride=stride,
            groups=expanded_c,
            norm_layer=norm_layer,
            activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c,
                                se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(
            expanded_c,
            out_planes=out_c,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Identity)  # 注意这里没有激活函数，传入Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):

    def __init__(self, kernel_size: int, input_c: int, out_c: int,
                 expand_ratio: int, stride: int, se_ratio: float,
                 drop_rate: float, norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNAct(
                input_c,
                expanded_c,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer)

            self.project_conv = ConvBNAct(
                expanded_c,
                out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity)  # 注意没有激活函数
        else:
            # 当只有project_conv时的情况
            self.project_conv = ConvBNAct(
                input_c,
                out_c,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer)  # 注意有激活函数

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result


@MODELS.register_module()
class EfficientNetV2(BaseBackbone):
    # repeat, kernel, stride, expansion, in_c, out_c, se_ratio, operator
    arch_settings = {
        'base': [[1, 3, 1, 1, 16, 16, 0, 0], [2, 3, 2, 4, 16, 32, 0, 0],
                 [2, 3, 2, 4, 32, 48, 0, 0], [3, 3, 2, 4, 48, 96, 0.25, 1],
                 [5, 3, 1, 6, 96, 112, 0.25, 1],
                 [8, 3, 2, 6, 112, 192, 0.25, 1]],
        's': [[2, 3, 1, 1, 24, 24, 0, 0], [4, 3, 2, 4, 24, 48, 0, 0],
              [4, 3, 2, 4, 48, 64, 0, 0], [6, 3, 2, 4, 64, 128, 0.25, 1],
              [9, 3, 1, 6, 128, 160, 0.25, 1],
              [15, 3, 2, 6, 160, 256, 0.25, 1]],
        'm': [[3, 3, 1, 1, 24, 24, 0, 0], [5, 3, 2, 4, 24, 48, 0, 0],
              [5, 3, 2, 4, 48, 80, 0, 0], [7, 3, 2, 4, 80, 160, 0.25, 1],
              [14, 3, 1, 6, 160, 176, 0.25,
               1], [18, 3, 2, 6, 176, 304, 0.25, 1],
              [5, 3, 1, 6, 304, 512, 0.25, 1]],
        'l': [[4, 3, 1, 1, 32, 32, 0, 0], [7, 3, 2, 4, 32, 64, 0, 0],
              [7, 3, 2, 4, 64, 96, 0, 0], [10, 3, 2, 4, 96, 192, 0.25, 1],
              [19, 3, 1, 6, 192, 224, 0.25,
               1], [25, 3, 2, 6, 224, 384, 0.25, 1],
              [7, 3, 1, 6, 384, 640, 0.25, 1]],
        'xl': [[4, 3, 1, 1, 32, 32, 0, 0], [8, 3, 2, 4, 32, 64, 0, 0],
               [8, 3, 2, 4, 64, 96, 0, 0], [16, 3, 2, 4, 96, 192, 0.25, 1],
               [24, 3, 1, 6, 192, 256, 0.25, 1],
               [32, 3, 2, 6, 256, 512, 0.25, 1],
               [8, 3, 1, 6, 512, 640, 0.25, 1]],
    }

    def __init__(self,
                 model_cnf: str = 's',
                 num_features: int = 1280,
                 drop_connect_rate: float = 0.0,
                 frozen_stages: int = 0,
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

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.layers = nn.ModuleList()
        model_cnf = self.arch_settings[model_cnf]

        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        stem_filter_num = model_cnf[0][4]
        self.layers.append(
            ConvBNAct(
                3,
                stem_filter_num,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer))  # active function default to SiLU

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0

        for idx,cnf in enumerate(model_cnf):
            blocks = []
            repeats = cnf[0]
            op = FusedMBConv if cnf[-1] == 0 else MBConv
            for i in range(repeats):
                blocks.append(
                    op(kernel_size=cnf[1],
                       input_c=cnf[4] if i == 0 else cnf[5],
                       out_c=cnf[5],
                       expand_ratio=cnf[3],
                       stride=cnf[2] if i == 0 else 1,
                       se_ratio=cnf[-2],
                       drop_rate=drop_connect_rate * block_id / total_blocks,
                       norm_layer=norm_layer))
                block_id += 1
            self.layers.append(Sequential(*blocks))

        head_input_c = model_cnf[-1][-3]
        self.layers.append(
            ConvBNAct(
                head_input_c,
                num_features,
                kernel_size=1,
                norm_layer=norm_layer))

        # head = OrderedDict()
        # num_classes = 1000
        # head.update({"project_conv": ConvBNAct(head_input_c,
        #                                        num_features,
        #                                        kernel_size=1,
        #                                        norm_layer=norm_layer)})
        # 激活函数默认是SiLU
        #
        # head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        # head.update({"flatten": nn.Flatten()})
        #
        # if dropout_rate > 0:
        #     head.update({"dropout": nn.Dropout(p=dropout_rate,
        #                   inplace=True)})
        # head.update({"classifier": nn.Linear(num_features, num_classes)})
        #
        # self.head = nn.Sequential(head)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
        # x = self.head(x)
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

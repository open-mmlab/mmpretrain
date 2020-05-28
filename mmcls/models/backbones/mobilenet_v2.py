import logging

import torch.nn as nn
import torch.utils.checkpoint as cp

from ..runner import load_checkpoint
from .base_backbone import BaseBackbone
from .weight_init import constant_init, kaiming_init


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


def conv_1x1_bn(inp, oup, act=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        act(inplace=True)
    )


class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 activation=nn.ReLU6):
        padding = (kernel_size - 1) // 2

        try:
            self.activation = activation(inplace=True)
        except RuntimeWarning('inplace is not allowed to use'):
            self.activation = activation()

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            self.activation
        )


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 stride,
                 expand_ratio,
                 activation=nn.ReLU6,
                 with_cp=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and inplanes == outplanes
        hidden_dim = int(round(inplanes * expand_ratio))

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inplanes,
                                     hidden_dim,
                                     kernel_size=1,
                                     activation=activation))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim,
                       hidden_dim,
                       stride=stride,
                       groups=hidden_dim,
                       activation=activation),
            # pw-linear
            nn.Conv2d(hidden_dim, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


def make_inverted_res_layer(block,
                            inplanes,
                            planes,
                            num_blocks,
                            stride=1,
                            expand_ratio=6,
                            activation_type=nn.ReLU6,
                            with_cp=False):
    layers = []
    for i in range(num_blocks):
        if i == 0:
            layers.append(block(inplanes, planes, stride,
                                expand_ratio=expand_ratio,
                                activation=activation_type,
                                with_cp=with_cp))
        else:
            layers.append(block(inplanes, planes, 1,
                                expand_ratio=expand_ratio,
                                activation=activation_type,
                                with_cp=with_cp))
    return nn.Sequential(*layers)


class MobileNetv2(BaseBackbone):
    """MobileNetv2 backbone.

    Args:
        widen_factor (float): Config of widen_factor.
        activation (str): Activation type of the network.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self,
                 widen_factor=1.,
                 activation=nn.ReLU6,
                 out_indices=(0, 1, 2, 3, 4, 5, 6),
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 with_cp=False):
        super(MobileNetv2, self).__init__()
        block = InvertedResidual

        inverted_residual_setting = {
            # lager_index: [expand_ratio, out_channel, n, stide]
            0: [1, 16, 1, 1],
            1: [6, 24, 2, 2],
            2: [6, 32, 3, 2],
            3: [6, 64, 4, 2],
            4: [6, 96, 3, 1],
            5: [6, 160, 3, 2],
            6: [6, 320, 1, 1]
        }
        self.widen_factor = widen_factor
        self.activation_type = activation
        try:
            self.activation = activation(inplace=True)
        except RuntimeWarning('inplace is not allowed to use'):
            self.activation = activation()

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.with_cp = with_cp

        self.inplanes = 32
        self.inplanes = _make_divisible(self.inplanes * widen_factor, 8)
        self.conv1 = conv3x3(3, self.inplanes, stride=2)

        self.inverted_res_layers = []
        for i, later_cfg in enumerate(inverted_residual_setting):
            t, c, n, s = later_cfg
            planes = _make_divisible(c * widen_factor, 8)
            inverted_res_layer = make_inverted_res_layer(
                block,
                self.inplanes,
                planes,
                num_blocks=n,
                stride=s,
                expand_ratio=t,
                activation_type=self.activation_type,
                with_cp=self.with_cp)
            self.inplanes = planes
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, inverted_res_layer)
            self.inverted_res_layers.append(layer_name)

        self.out_channel = 1280
        self.out_channel = int(self.out_channel * widen_factor) \
            if widen_factor > 1.0 else self.out_channel
        self.conv1_bn = conv_1x1_bn(self.inplanes, self.out_channel)

        self.feat_dim = self.out_channel

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        outs = []
        for i, layer_name in enumerate(self.inverted_res_layers):
            inverted_res_layer = getattr(self, layer_name)
            x = inverted_res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        x = self.conv1_bn(x)
        outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(MobileNetv2, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False

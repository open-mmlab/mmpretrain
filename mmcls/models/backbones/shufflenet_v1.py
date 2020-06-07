import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.runner import load_checkpoint

from .base_backbone import BaseBackbone
from .weight_init import constant_init, kaiming_init


def conv3x3(inplanes, planes, stride=1, padding=1, bias=False, groups=1):
    """3x3 convolution with padding
    """
    return nn.Conv2d(
        inplanes,
        planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def conv1x1(inplanes, planes, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution when groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(inplanes, planes, kernel_size=1, groups=groups, stride=1)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ShuffleUnit(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 groups=3,
                 first_block=True,
                 combine='add',
                 with_cp=False):

        super(ShuffleUnit, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.first_block = first_block
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.planes // 4
        self.with_cp = with_cp

        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.planes -= self.inplanes
        else:
            raise ValueError("Cannot combine tensors with \"{}\" "
                             "Only \"add\" and \"concat\" are "
                             "supported".format(self.combine))

        if combine == 'add':
            assert inplanes == planes, \
                'inplanes must be equal to outplanes when combine is add'

        self.first_1x1_groups = self.groups if first_block else 1
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.inplanes,
            self.bottleneck_channels,
            self.first_1x1_groups,
            batch_norm=True,
            relu=True)

        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels,
            self.bottleneck_channels,
            stride=self.depthwise_stride,
            groups=self.bottleneck_channels)
        self.bn_after_depthwise = \
            nn.BatchNorm2d(self.bottleneck_channels)

        self.g_conv_1x1_expand = \
            self._make_grouped_conv1x1(self.bottleneck_channels,
                                       self.planes,
                                       self.groups,
                                       batch_norm=True,
                                       relu=False)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    @staticmethod
    def _make_grouped_conv1x1(inplanes,
                              planes,
                              groups,
                              batch_norm=True,
                              relu=False):

        modules = OrderedDict()

        conv = conv1x1(inplanes, planes, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(planes)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            if self.combine == 'concat':
                residual = self.avgpool(residual)

            out = self.g_conv_1x1_compress(x)
            out = channel_shuffle(out, self.groups)
            out = self.depthwise_conv3x3(out)
            out = self.bn_after_depthwise(out)
            out = self.g_conv_1x1_expand(out)

            out = self._combine_func(residual, out)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ShuffleNetv1(BaseBackbone):
    """ShuffleNetv1 backbone.

    Args:
        groups (int): number of groups to be used in grouped
            1x1 convolutions in each ShuffleUnit. Default is 3 for best
            performance according to original paper.
        widen_factor (float): Config of widen_factor.
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
                 groups=3,
                 widen_factor=1.0,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 with_cp=False):
        super(ShuffleNetv1, self).__init__()
        blocks = [3, 7, 3]
        self.groups = groups
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.with_cp = with_cp

        if groups == 1:
            channels = [144, 288, 576]
        elif groups == 2:
            channels = [200, 400, 800]
        elif groups == 3:
            channels = [240, 480, 960]
        elif groups == 4:
            channels = [272, 544, 1088]
        elif groups == 8:
            channels = [384, 768, 1536]
        else:
            raise ValueError("{} groups is not supported for "
                             "1x1 Grouped Convolutions".format(groups))
        channels = [_make_divisible(ch * widen_factor, 8) for ch in channels]

        self.inplanes = int(24 * widen_factor)
        self.conv1 = conv3x3(3, self.inplanes, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            channels[0], blocks[0], first_block=False, with_cp=with_cp)
        self.layer2 = self._make_layer(channels[1], blocks[1], with_cp=with_cp)
        self.layer3 = self._make_layer(channels[2], blocks[2], with_cp=with_cp)

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

    def _make_layer(self, outplanes, blocks, first_block=True, with_cp=False):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(
                    ShuffleUnit(
                        self.inplanes,
                        outplanes,
                        groups=self.groups,
                        first_block=first_block,
                        combine='concat',
                        with_cp=with_cp))
            else:
                layers.append(
                    ShuffleUnit(
                        self.inplanes,
                        outplanes,
                        groups=self.groups,
                        first_block=True,
                        combine='add',
                        with_cp=with_cp))
            self.inplanes = outplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        x = self.layer1(x)
        if 0 in self.out_indices:
            outs.append(x)
        x = self.layer2(x)
        if 1 in self.out_indices:
            outs.append(x)
        x = self.layer3(x)
        if 2 in self.out_indices:
            outs.append(x)

        outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ShuffleNetv1, self).train(mode)
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
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False

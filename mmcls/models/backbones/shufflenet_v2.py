import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.runner import load_checkpoint

from .base_backbone import BaseBackbone
from .weight_init import constant_init, kaiming_init


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


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


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, with_cp=False):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 2):
            raise ValueError('illegal stride value')
        self.stride = stride
        self.with_cp = with_cp

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(
            i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):

        def _inner_forward(x):
            if self.stride == 1:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)
            else:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class ShuffleNetv2(BaseBackbone):
    """ShuffleNetv2 backbone.

    Args:
        groups (int): number of groups to be used in grouped
            1x1 convolutions in each ShuffleUnit. Default is 3 for best
            performance according to original paper.
        widen_factor (float): Config of widen_factor.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set nn.BatchNorm2d layers as eval mode,
            namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of
            nn.BatchNorm2d layers.
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
        super(ShuffleNetv2, self).__init__()
        blocks = [4, 8, 4]
        self.groups = groups
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.with_cp = with_cp

        if widen_factor == 0.5:
            channels = [48, 96, 192, 1024]
        elif widen_factor == 1.0:
            channels = [116, 232, 464, 1024]
        elif widen_factor == 1.5:
            channels = [176, 352, 704, 1024]
        elif widen_factor == 2.0:
            channels = [244, 488, 976, 2048]
        else:
            raise ValueError("""{} groups is not supported for
                1x1 Grouped Convolutions""".format(groups))

        self.inplanes = channels[0]
        self.conv1 = conv_bn(3, self.inplanes, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(channels[1], blocks[0], with_cp=with_cp)
        self.layer2 = self._make_layer(channels[2], blocks[1], with_cp=with_cp)
        self.layer3 = self._make_layer(channels[3], blocks[2], with_cp=with_cp)

        self.conv_out = conv_1x1_bn(self.inplanes, channels[-1])

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

    def _make_layer(self, outplanes, blocks, with_cp):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(
                    InvertedResidual(
                        self.inplanes, outplanes, stride=2, with_cp=with_cp))
            else:
                layers.append(
                    InvertedResidual(
                        self.inplanes, outplanes, stride=1, with_cp=with_cp))
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

        x = self.conv_out(x)
        outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ShuffleNetv2, self).train(mode)
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

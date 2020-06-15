import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      constant_init, kaiming_init)
from torch.nn.modules.batchnorm import _BatchNorm

from .base_backbone import BaseBackbone


def channel_shuffle(x, groups):
    """ Channel Shuffle operation.

    This function enables cross-group information flow for multiple group
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batchsize, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x


def make_divisible(value, divisor, min_value=None):
    """ Make divisible function.

    This function ensures that all layers have a channel number that is
    divisible by divisor.

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): the minimum value of the output channel.

    Returns:
        int: The modified output channel number
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ShuffleUnit(nn.Module):
    """ShuffleUnit block.

    ShuffleNet unit with pointwise group convolution (GConv) and channel
    shuffle.

    Args:
        inplanes (int): The input channels of the ShuffleUnit.
        planes (int): The output channels of the ShuffleUnit.
        groups (int, optional): The number of groups to be used in grouped 1x1
            convolutions in each ShuffleUnit. Default: 3
        first_block (bool, optional): Whether is the first ShuffleUnit of a
            sequential ShuffleUnits. Default: True, which means using the
            grouped 1x1 convolution.
        combine (str, optional): The ways to combine the input and output
            branches. Default: 'add'.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 inplanes,
                 planes,
                 groups=3,
                 first_block=True,
                 combine='add',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
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
            assert inplanes == planes, ('inplanes must be equal to '
                                        'planes when combine is add.')
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.planes -= self.inplanes
        else:
            raise ValueError(f'Cannot combine tensors with {self.combine}. '
                             'Only "add" and "concat" are supported.')

        self.first_1x1_groups = self.groups if first_block else 1
        self.g_conv_1x1_compress = ConvModule(
            in_channels=self.inplanes,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            groups=self.first_1x1_groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.depthwise_conv3x3_bn = ConvModule(
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=3,
            stride=self.depthwise_stride,
            padding=1,
            groups=self.bottleneck_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.g_conv_1x1_expand = ConvModule(
            in_channels=self.bottleneck_channels,
            out_channels=self.planes,
            kernel_size=1,
            groups=self.groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.act = build_activation_layer(act_cfg)

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            if self.combine == 'concat':
                residual = self.avgpool(residual)

            out = self.g_conv_1x1_compress(x)
            out = channel_shuffle(out, self.groups)
            out = self.depthwise_conv3x3_bn(out)
            out = self.g_conv_1x1_expand(out)

            out = self._combine_func(residual, out)
            out = self.act(out)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class ShuffleNetv1(BaseBackbone):
    """ShuffleNetv1 backbone.

    Args:
        groups (int, optional): The number of groups to be used in grouped 1x1
            convolutions in each ShuffleUnit. Default is 3 for best performance
            according to original paper. Default: 3.
        widen_factor (float, optional): Width multiplier - adjusts number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3)
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 groups=3,
                 widen_factor=1.0,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False):
        super(ShuffleNetv1, self).__init__()
        self.stage_blocks = [3, 7, 3]
        self.groups = groups

        for indice in out_indices:
            if indice not in range(0, 4):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 4). But received {indice}')

        if frozen_stages not in range(-1, 4):
            raise ValueError('frozen_stages must in range(-1, 4). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        if groups == 1:
            channels = (144, 288, 576)
        elif groups == 2:
            channels = (200, 400, 800)
        elif groups == 3:
            channels = (240, 480, 960)
        elif groups == 4:
            channels = (272, 544, 1088)
        elif groups == 8:
            channels = (384, 768, 1536)
        else:
            raise ValueError(f'{groups} groups is not supported for 1x1 '
                             'Grouped Convolutions')

        channels = [make_divisible(ch * widen_factor, 8) for ch in channels]

        self.inplanes = int(24 * widen_factor)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels=3,
            out_channels=self.inplanes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            first_block = False if i == 0 else True
            layer = self.make_layer(channels[i], num_blocks, first_block)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None. But received '
                            f'{type(pretrained)}')

    def make_layer(self, planes, num_blocks, first_block=True):
        """ Stack ShuffleUnit blocks to make a layer.

        Args:
            planes (int): planes of block.
            num_blocks (int): number of blocks.
            first_block (bool, optional): Whether is the first ShuffleUnit of a
                sequential ShuffleUnits. Default: True, which means usingf the
                grouped 1x1 convolution.
        """
        layers = []
        for i in range(num_blocks):
            first_block = first_block if i == 0 else True
            combine_mode = 'concat' if i == 0 else 'add'
            layers.append(
                ShuffleUnit(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    first_block=first_block,
                    combine=combine_mode,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ShuffleNetv1, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

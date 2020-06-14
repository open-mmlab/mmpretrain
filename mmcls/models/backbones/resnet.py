import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class BasicBlock(nn.Module):
    """BasicBlock for ResNet.

    Args:
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicBlock, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


@BACKBONES.register_module()
class ResNet(BaseBackbone):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        base_channels (int): Number of base channels of hidden layer.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmcls.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = base_channels

        self._make_stem_layer(in_channels, base_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, base_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    base_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, base_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    base_channels // 2,
                    base_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, base_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    base_channels // 2,
                    base_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, base_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, base_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        super(ResNet, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class ResNetV1d(ResNet):
    """ResNetV1d variant described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv
    in the input stem with three 3x3 convs. And in the downsampling block,
    a 2x2 avg_pool with stride 2 is added before conv, whose stride is
    changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)

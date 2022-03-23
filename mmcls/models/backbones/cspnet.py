# Copyright (c) OpenMMLab. All rights reserved.
import math
from abc import ABCMeta, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule, Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from .resnet import Bottleneck as ResNetBottleneck
from .resnext import Bottleneck as ResNeXtBottleneck

eps = 1.0e-5


class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet. Each DarknetBottleneck
    consists of two ConvModules and the input is added to the final output.
    Each ConvModule is composed of Conv, BN, and LeakyReLU. The first convLayer
    has filter size of 1x1 and the second one has the filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        drop_path_rate (float): The ratio of the drop path layer. Default: 0.
        norm_cfg (dict): Config dict for normalization layer.
            Default:dict(type='BN', eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=2,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 drop_path_rate=0,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels / expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop_path(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPStage(BaseModule):
    """Cross Stage Partial Stage.

    CrossStage = downsamper_conv(x) + expand_conv(x) + split(x)->[xa,xb]
    + blocks(xb) + transition_conv(xb) + concat(xa, xb) + final_conv(x)

    Args:
        block (nn.module): The basic block function in the Stage.
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        has_downsamper (bool): Whether to add a downsamper in the stage.
            Default: False.
        down_growth (bool): Whether to expand the channels in the
            downsamper layer of the stage. Default: False.
        expand_ratio (float): The expand ratio to adjust the number of
             channels of the expand conv layer. Default: 0.5
        bottle_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        drop_path_rate (float): The ratio of the drop path layer in the
            blocks of the stage. Default: 0.
        num_blocks (int): Number of blocks. Default: 1
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', inplace=True)
    """

    def __init__(self,
                 block,
                 in_channels,
                 out_channels,
                 has_downsamper=True,
                 down_growth=False,
                 expand_ratio=0.5,
                 bottle_ratio=2,
                 num_blocks=1,
                 block_dpr=0,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        # grow downsample channels to output channels
        down_channels = out_channels if down_growth else in_channels

        if has_downsamper:
            self.downsamper_conv = ConvModule(
                in_channels=in_channels,
                out_channels=down_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=32 if block is ResNeXtBottleneck else 1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.downsamper_conv = nn.Identity()

        exp_channels = int(down_channels * expand_ratio)
        self.expand_conv = ConvModule(
            in_channels=down_channels,
            out_channels=exp_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg if block is DarknetBottleneck else None)

        assert exp_channels % 2 == 0, \
            'The channel number before blocks must be divisible by 2.'
        block_channcels = exp_channels // 2
        block_cfg = dict(
            in_channels=block_channcels,
            out_channels=block_channcels,
            expansion=bottle_ratio,
            drop_path_rate=block_dpr,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # there is a extra attribute base_channels in ResNeXtBottleneck
        # the base_channels change from 64 to 32 in CSPNet
        if block is ResNeXtBottleneck:
            block_cfg['base_channels'] = 32
        self.blocks = nn.Sequential(
            *[block(**block_cfg) for _ in range(num_blocks)])
        self.atfer_blocks_conv = ConvModule(
            block_channcels,
            block_channcels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.final_conv = ConvModule(
            2 * block_channcels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.downsamper_conv(x)
        x = self.expand_conv(x)

        split = x.shape[1] // 2
        xa, xb = x[:, :split], x[:, split:]

        xb = self.blocks(xb)
        xb = self.atfer_blocks_conv(xb).contiguous()

        x_final = torch.cat((xa, xb), dim=1)
        return self.final_conv(x_final)


class CSPNet(BaseModule, metaclass=ABCMeta):
    """base CSPNet for CSPDarkNet, CSPResNet and CSPResNeXt.

    CrossStage = downsamper_conv(x) + expand_conv(x) + split(x)->[xa,xb]
    + blocks(xb) + transition_conv(xb) + concat(xa, xb) + final_conv(x)

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 out_indices=(4, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 norm_eval=False,
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super().__init__(init_cfg)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

        assert hasattr(self, 'arch_setting'), \
            'Please set arch_setting attribute in ' \
            f'definition of Class : {self.__class__}.'

        self._make_stem_layer(in_channels)

        stages = []
        for stage_setting in self.arch_setting:
            (block_fn, in_channels, out_channels, num_blocks, expand_ratio,
             bottle_ratio, has_downsamper, down_growth) = stage_setting
            csp_stage = CSPStage(
                block_fn,
                in_channels,
                out_channels,
                num_blocks=num_blocks,
                expand_ratio=expand_ratio,
                bottle_ratio=bottle_ratio,
                has_downsamper=has_downsamper,
                down_growth=down_growth,
                block_dpr=0,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                init_cfg=init_cfg)
            stages.append(csp_stage)
        self.stages = Sequential(*stages)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = len(self.stages) + index
            assert 0 <= out_indices[i] <= len(self.stages), \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

    @abstractmethod
    def _make_stem_layer(self, in_channels):
        pass

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(CSPNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []

        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class CSPDarkNet(CSPNet):
    """CSP-Darknet backbone used in YOLOv4.

    Args:
        depth (int): Depth of CSP-Darknet. Default: 53.
        in_channels (int): Number of input image channels. Default: 3.
        out_indices (Sequence[int]): Output from which stages.
            Default: (3, ).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmcls.models import CSPDarkNet
        >>> import torch
        >>> model = CSPDarkNet(depth=53, out_indices=(0, 1, 2, 3, 4))
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 64, 208, 208)
        (1, 128, 104, 104)
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right: [block_fn, in_channels, out_channels, num_blocks,
    # expand_ratio, bottle_ratio, has_downsamper, down_growth] in CSPStage.
    arch_settings = {
        53: [[DarknetBottleneck, 32, 64, 1, 2, 2, True, True],
             [DarknetBottleneck, 64, 128, 2, 1, 1, True, True],
             [DarknetBottleneck, 128, 256, 8, 1, 1, True, True],
             [DarknetBottleneck, 256, 512, 8, 1, 1, True, True],
             [DarknetBottleneck, 512, 1024, 4, 1, 1, True, True]],
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 out_indices=(4, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):

        assert depth in self.arch_settings, 'depth must be one of ' \
            f'{list(self.arch_settings.keys())}, but get {depth}.'
        self.arch_setting = self.arch_settings[depth]
        if frozen_stages not in range(-1, len(self.arch_setting)):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting)). But received '
                             f'{frozen_stages}')

        super().__init__(
            in_channels=in_channels,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def _make_stem_layer(self, in_channels):
        """using a stride=1 conv as the stem in CSPDarknet."""
        # `stem_channels` equals to the `in_channels` in the first stage.
        stem_channels = self.arch_setting[0][1]
        self.stem = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


@BACKBONES.register_module()
class CSPResNet(CSPNet):
    """CSP-ResNet backbone.

    Args:
        depth (int): Depth of CSP-ResNet. Default: 50.
        out_indices (Sequence[int]): Output from which stages.
            Default: (4, ).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmcls.models import CSPResNet
        >>> import torch
        >>> self = CSPResNet(depth=50, out_indices=(0, 1, 2, 3))
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 128, 104, 104)
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right: [block_fn, in_channels, out_channels, num_blocks,
    # expand_ratio, bottle_ratio, has_downsamper, down_growth] in CSPStage.
    arch_settings = {
        50: [[ResNetBottleneck, 64, 128, 3, 4, 2, False, False],
             [ResNetBottleneck, 128, 256, 3, 4, 2, True, False],
             [ResNetBottleneck, 256, 512, 5, 4, 2, True, False],
             [ResNetBottleneck, 512, 1024, 2, 4, 2, True, False]],
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 deep_stem=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 norm_eval=False,
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        assert depth in self.arch_settings, 'depth must be one of ' \
            f'{list(self.arch_settings.keys())}, but get {depth}.'
        self.arch_setting = self.arch_settings[depth]
        self.frozen_stages = frozen_stages
        if self.frozen_stages not in range(-1, len(self.arch_setting)):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting)). But received '
                             f'{self.frozen_stages}')
        self.deep_stem = deep_stem

        super().__init__(
            in_channels=in_channels,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def _make_stem_layer(self, in_channels):
        # `stem_channels` equals to the `in_channels` in the first stage.
        stem_channels = self.arch_setting[0][1]
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        else:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


@BACKBONES.register_module()
class CSPResNeXt(CSPResNet):
    """CSP-ResNeXt backbone.

    Args:
        depth (int): Depth of CSP-ResNeXt. Default: 50.
        out_indices (Sequence[int]): Output from which stages.
            Default: (4, ).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmcls.models import CSPResNeXt
        >>> import torch
        >>> model = CSPResNeXt(depth=50, out_indices=(0, 1, 2, 3))
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 56, 56)
        (1, 512, 28, 28)
        (1, 1024, 14, 14)
        (1, 2048, 7, 7)
    """
    # From left to right: [block_fn, in_channels, out_channels, num_blocks,
    # expand_ratio, bottle_ratio, has_downsamper, down_growth] in CSPStage.
    arch_settings = {
        50: [
            [ResNeXtBottleneck, 64, 256, 3, 4, 4, False, False],
            [ResNeXtBottleneck, 256, 512, 3, 2, 4, True, False],
            [ResNeXtBottleneck, 512, 1024, 5, 2, 4, True, False],
            [ResNeXtBottleneck, 1024, 2048, 2, 2, 4, True, False],
        ],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

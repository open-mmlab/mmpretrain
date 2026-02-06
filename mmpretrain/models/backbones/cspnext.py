# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Sequence, Union

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmpretrain.registry import MODELS
from mmpretrain.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig, MultiConfig
#from ..layers import SPPFBottleneck
from mmengine.model import BaseModule
from .base_backbone_yolo import BaseBackbone

# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Sequence, Union

from mmengine.model import BaseModule

from ..layers.yolo_blocks import DarknetBottleneck, CSPLayer, SPPFBottleneck

@MODELS.register_module()
class CSPNeXt(BaseBackbone):
    """CSPNeXt backbone used in RTMDet.

    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
            Defaults to P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.Defaults to
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        plugins: Union[dict, List[dict]] = None,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_ovewrite: dict = None,
        channel_attention: bool = True,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN'),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        norm_eval: bool = False,
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        self.channel_attention = channel_attention
        self.use_depthwise = use_depthwise
        self.conv = DepthwiseSeparableConvModule \
            if use_depthwise else ConvModule
        self.expand_ratio = expand_ratio
        self.conv_cfg = conv_cfg

        super().__init__(
            arch_setting,
            deepen_factor,
            widen_factor,
            input_channels,
            out_indices,
            frozen_stages=frozen_stages,
            plugins=plugins,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        stem = nn.Sequential(
            ConvModule(
                3,
                int(self.arch_setting[0][0] * self.widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                int(self.arch_setting[0][0] * self.widen_factor // 2),
                int(self.arch_setting[0][0] * self.widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                int(self.arch_setting[0][0] * self.widen_factor // 2),
                int(self.arch_setting[0][0] * self.widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)
        num_blocks = max(round(num_blocks * self.deepen_factor), 1)

        stage = []
        conv_layer = self.conv(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        csp_layer = CSPLayer(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            use_depthwise=self.use_depthwise,
            # use_cspnext_block=True,
            expand_ratio=self.expand_ratio,
            # channel_attention=self.channel_attention,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)
        return stage

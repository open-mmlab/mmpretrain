# Copyright (c) OpenMMLab. All rights reserved.
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from mmcv.runner.base_module import ModuleList, Sequential

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm that supports two data formats: channels_last (default) or
    channels_first.

    The ordering of the dimensions in the inputs. channels_last corresponds to
    inputs with shape (batch_size, height, width, channels) while
    channels_first corresponds to inputs with shape (batch_size, channels,
    height, width).
    """

    def __init__(self,
                 num_channels: int,
                 dim: int = 1,
                 eps: float = 1e-6,
                 **kwargs) -> None:
        super().__init__(num_channels, eps=eps, **kwargs)
        self.num_channels = self.normalized_shape[0]
        self.dim = dim

    def forward(self, x):
        if self.dim == -1 or self.dim == x.dim() - 1:
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        else:
            u = x.mean(self.dim, keepdim=True)
            s = (x - u).pow(2).mean(self.dim, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)

            weight_shape = [1] * self.dim + [self.num_channels] + [1] * (
                x.dim() - self.dim - 1)
            x = self.weight.view(weight_shape) * x + self.bias.view(
                weight_shape)
            return x


class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        channels_last (bool): Whether to permute the channel's dim to the last
            before the layer norm.More details can be found in the note.
            Defaults to True.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU
           -> 1x1 Conv; all outputs are in (N, C, H, W).
        2. DwConv -> Permute to (N, H, W, C) -> LayerNorm (channels_last)
           -> Linear -> GELU -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster because LayerNorm on the last dim is
        officially supported by PyTorch.
    """

    def __init__(self,
                 in_channels,
                 drop_path_rate=0.,
                 channels_last=True,
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels)

        self.channels_last = channels_last
        self.norm = LayerNorm2d(in_channels, dim=-1 if channels_last else 1)
        # Use linear layer to do pointwise conv.
        self.pointwise_conv1 = nn.Linear(in_channels, 4 * in_channels)
        self.act = nn.GELU()
        self.pointwise_conv2 = nn.Linear(4 * in_channels, in_channels)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)

        if self.channels_last:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)
        return x


@BACKBONES.register_module()
class ConvNeXt(BaseBackbone):
    """ConvNeXt.

    A PyTorch implementation of : `A ConvNet for the 2020s
    <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        init_cfg (dict, optional): Initialization config dict
    """
    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        self.depths = arch['depths']
        self.channels = arch['channels']
        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=4, stride=4),
            LayerNorm2d(self.channels[0], dim=1),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1], dim=1),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = LayerNorm2d(channels, dim=1)
                self.add_module(f'norm{i}', norm_layer)

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    outs.append(norm_layer(x.mean([-2, -1])))
                else:
                    outs.append(norm_layer(x))

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()

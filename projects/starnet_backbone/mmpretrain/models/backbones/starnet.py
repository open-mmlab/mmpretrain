from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmcv.cnn.bricks import (ConvModule, DropPath, build_activation_layer,
                             build_norm_layer)
from mmengine.model import BaseModule, Sequential

from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.registry import MODELS


class Block(BaseModule):
    """StarNet Block.

    Args:
        in_channels (int): The number of input channels.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 3.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='ReLU6')``.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to ``dict(type='Conv2d')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels,
        mlp_ratio: float = 3.,
        drop_path: float = 0.,
        conv_cfg: Optional[dict] = dict(type='Conv2d'),
        norm_cfg: Optional[dict] = dict(type='BN'),
        act_cfg: Optional[dict] = dict(type='ReLU6'),
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.dwconv = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=7,
            stride=1,
            padding=(7 - 1) // 2,
            groups=in_channels,
            bias=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        self.fc1 = ConvModule(
            in_channels=in_channels,
            out_channels=mlp_ratio * in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
        )

        self.fc2 = ConvModule(
            in_channels=in_channels,
            out_channels=mlp_ratio * in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
        )
        self.g = ConvModule(
            in_channels=mlp_ratio * in_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        self.dwconv2 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=7,
            stride=1,
            padding=(7 - 1) // 2,
            groups=in_channels,
            bias=True,
            conv_cfg=conv_cfg,
            norm_cfg=None,
        )
        self.act = build_activation_layer(act_cfg)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x1, x2 = self.fc1(x), self.fc2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = identity + self.drop_path(x)
        return x


@MODELS.register_module()
class StarNet(BaseBackbone):
    """StarNet.

    A PyTorch implementation of StarNet introduced by:
    `Rewrite the Stars <https://arxiv.org/abs/2403.19967>`_

    Modified from the `official repo
    <https://github.com/ma-xu/Rewrite-the-Stars?tab=readme-ov-file>`.

    Args:
        arch (str | dict): The model's architecture.
            it should include the following two keys:

            - layers (list[int]): Number of blocks at each stage.
            - embed_dims (list[int]): The number of channels at each stage.

            Defaults to 's1'.

        in_channels (int): Number of input image channels. Default: 3.
        out_channels (int): Output channels of the stem layer. Default: 32.
        mlp_ratio (float): The expansion ratio in pointwise convolution.
            Defaults to 4.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='ReLU6')``.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to ``dict(type='Conv2d')``.
        init_cfg (dict, optional): Initialization config dict
    """

    arch_settings = {
        's1': {
            'layers': [2, 2, 8, 3],
            'embed_dims': [24, 48, 96, 192],
        },
        's2': {
            'layers': [1, 2, 6, 2],
            'embed_dims': [32, 64, 128, 256],
        },
        's3': {
            'layers': [2, 2, 8, 4],
            'embed_dims': [32, 64, 128, 256],
        },
        's4': {
            'layers': [3, 3, 12, 5],
            'embed_dims': [32, 64, 128, 256],
        },
        's050': {
            'layers': [1, 1, 3, 1],
            'embed_dims': [16, 32, 64, 128],
        },
        's100': {
            'layers': [1, 2, 4, 1],
            'embed_dims': [20, 40, 80, 160],
        },
        's150': {
            'layers': [1, 2, 4, 2],
            'embed_dims': [24, 48, 96, 192],
        }
    }

    def __init__(
        self,
        arch='s1',
        in_channels: int = 3,
        out_channels: int = 32,
        out_indices=-1,
        frozen_stages=0,
        mlp_ratio: float = 4.,
        drop_path_rate: float = 0.,
        conv_cfg: Optional[dict] = dict(type='Conv2d'),
        norm_cfg: Optional[dict] = dict(type='BN'),
        act_cfg: Optional[dict] = dict(type='ReLU6'),
        init_cfg=[
            dict(type='Kaiming', layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm'])
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "layers" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'

        self.layers = arch['layers']
        self.embed_dims = arch['embed_dims']
        depth = len(self.layers)
        self.num_stages = len(self.layers)
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stem = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.layers))
        ]

        self.stages = []
        cur = 0
        for i in range(depth):
            stage = self._make_stage(
                planes=self.out_channels,
                num_blocks=self.layers[i],
                cur=cur,
                dpr=dpr,
                stages_num=i)
            self.out_channels = self.embed_dims[i]
            cur += self.layers[i]
            stage_name = f'stage{i}'
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
            assert 0 <= out_indices[i] <= self.num_stages, \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

        if self.out_indices:
            for i_layer in self.out_indices:
                layer = build_norm_layer(norm_cfg, self.embed_dims[i_layer])[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _make_stage(self, planes, num_blocks, cur, dpr, stages_num):
        down_sampler = ConvModule(
            in_channels=planes,
            out_channels=self.embed_dims[stages_num],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
        )

        blocks = [
            Block(
                in_channels=self.embed_dims[stages_num],
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[cur + i],
            ) for i in range(num_blocks)
        ]

        return Sequential(down_sampler, *blocks)

    def forward(self, x):
        x = self.stem(x)

        outs = []
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            x = stage(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            stage_layer = getattr(self, f'stage{i}')
            stage_layer.eval()

            for param in stage_layer.parameters():
                param.requires_grad = False
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(StarNet, self).train(mode)
        self._freeze_stages()

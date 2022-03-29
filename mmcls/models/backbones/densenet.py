# Copyright (c) OpenMMLab. All rights reserved.
import math
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import build_activation_layer, build_norm_layer
from torch.jit.annotations import List

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class DenseLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 growth_rate,
                 bn_size,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_rate=0.,
                 memory_efficient=False):
        super(DenseLayer, self).__init__()

        self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
        self.conv1 = nn.Conv2d(
            in_channels,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False)
        self.act = build_activation_layer(act_cfg)
        self.norm2 = build_norm_layer(norm_cfg, bn_size * growth_rate)[1]
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bottleneck_fn(self, xs):
        # type: (List[torch.Tensor]) -> torch.Tensor
        concated_features = torch.cat(xs, 1)
        bottleneck_output = self.conv1(
            self.act(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, x):
        # type: (List[torch.Tensor]) -> bool
        for tensor in x:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, x):
        # type: (List[torch.Tensor]) -> torch.Tensor
        def closure(*xs):
            return self.bottleneck_fn(xs)

        return cp.checkpoint(closure, *x)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):  # noqa: F811
        # type: (List[torch.Tensor]) -> (torch.Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):  # noqa: F811
        # type: (torch.Tensor) -> (torch.Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, x):  # noqa: F811
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception('Memory Efficient not supported in JIT')
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bottleneck_fn(prev_features)

        new_features = self.conv2(self.act(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.Module):

    def __init__(self,
                 num_layers,
                 in_channels,
                 bn_size,
                 growth_rate,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_rate=0.,
                 memory_efficient=False):
        super(DenseBlock, self).__init__()
        self.block = nn.ModuleList([
            DenseLayer(
                in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient) for i in range(num_layers)
        ])

    def forward(self, init_features):
        features = [init_features]
        for layer in self.block:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseTransition(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(DenseTransition, self).__init__()
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.act = build_activation_layer(act_cfg)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


@BACKBONES.register_module()
class DenseNet(BaseBackbone):

    arch_settings = {
        '121': {
            'growth_rate': 32,
            'depths': [6, 12, 24, 16],
            'init_channels': 64,
        },
        '161': {
            'growth_rate': 48,
            'depths': [6, 12, 36, 24],
            'init_channels': 96,
        },
        '169': {
            'growth_rate': 32,
            'depths': [6, 12, 32, 32],
            'init_channels': 64,
        },
        '201': {
            'growth_rate': 32,
            'depths': [6, 12, 48, 32],
            'init_channels': 64,
        },
    }

    def __init__(self,
                 arch='121',
                 in_channels=3,
                 bn_size=4,
                 drop_rate=0,
                 compression_factor=0.5,
                 memory_efficient=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 out_indices=-1,
                 frozen_stages=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            essential_keys = {'growth_rate', 'depths', 'init_channels'}
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'

        self.growth_rate = arch['growth_rate']
        self.depths = arch['depths']
        self.init_channels = arch['init_channels']
        self.act = build_activation_layer(act_cfg)

        self.num_stages = len(self.depths)

        # check out indices and frozen stages
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Set stem layers
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.init_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False),
            build_norm_layer(norm_cfg, self.init_channels)[1], self.act,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Repetitions of DenseNet Blocks
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        channels = self.init_channels
        for i in range(self.num_stages):
            depth = self.depths[i]

            stage = DenseBlock(
                num_layers=depth,
                in_channels=channels,
                bn_size=bn_size,
                growth_rate=self.growth_rate,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient)
            self.stages.append(stage)
            channels += depth * self.growth_rate

            if i != self.num_stages - 1:
                transition = DenseTransition(
                    in_channels=channels,
                    out_channels=math.floor(channels * compression_factor),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                channels = math.floor(channels * compression_factor)
            else:
                # Final layers after dense block is just bn with act.
                # Unlike the paper, the original repo also put this in
                # transition layer, whereas torchvision take this out.
                # We reckon this as transition layer here.
                transition = nn.Sequential(
                    build_norm_layer(norm_cfg, channels)[1],
                    self.act,
                )
            self.transitions.append(transition)

        self._freeze_stages()

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i in range(self.num_stages):
            x = self.stages[i](x)
            x = self.transitions[i](x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.transitions[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(DenseNet, self).train(mode)
        self._freeze_stages()

# Copyright (c) OpenMMLab. All rights reserved.x
import itertools
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Sequence, Union

from mmcv.cnn.bricks import ConvModule, DropPath, build_upsample_layer, build_conv_layer
from mmengine.model import BaseModule, Sequential
from mmengine.model.weight_init import trunc_normal_

from ..utils import LayerScale

from mmcls.registry import MODELS
from ..utils import build_norm_layer, to_2tuple
from .base_backbone import BaseBackbone

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.layers.helpers import to_2tuple

EfficientFormer_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}

EfficientFormer_depth = {
    'L': [5, 5, 15, 10],  # 26m 83.3%
    'S2': [4, 4, 12, 8],  # 12m
    'S1': [3, 3, 9, 6],  # 79.0
    'S0': [2, 2, 6, 4],  # 75.7
}

# 26m
expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

# 12m
expansion_ratios_S2 = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}

# 6.1m
expansion_ratios_S1 = {
    '0': [4, 4, 4],
    '1': [4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
    '3': [4, 4, 3, 3, 4, 4],
}

# 3.5m
expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}


class Attention4D(BaseModule):
    def __init__(self,
                 dim=384,
                 key_dim=32,
                 num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 stride=None,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 upsample_cfg=dict(type='bilinear'),
                 init_cfg=None):
        super(Attention4D, self).__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.N = self.resolution ** 2
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = ConvModule(in_channels=dim,
                                          out_channels=dim,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          groups=dim,
                                          bias=True,
                                          conv_cfg=conv_cfg,
                                          norm_cfg=norm_cfg,
                                          act_cfg=None)
            self.upsample = build_upsample_layer(upsample_cfg, scale_factor=stride)
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.q = ConvModule(in_channels=dim,
                            out_channels=self.num_heads * self.key_dim,
                            kernel_size=1,
                            bias=True,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=None)
        self.k = ConvModule(in_channels=dim,
                            out_channels=self.num_heads * self.key_dim,
                            kernel_size=1,
                            bias=True,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=None)
        self.v = ConvModule(in_channels=dim,
                            out_channels=self.num_heads * self.d,
                            kernel_size=1,
                            bias=True,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=None)

        self.v_local = ConvModule(in_channels=self.num_heads * self.d,
                                  out_channels=self.num_heads * self.d,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  groups=self.num_heads * self.d,
                                  bias=True,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=None)

        self.talking_head1 = build_conv_layer(conv_cfg,
                                              self.num_heads,
                                              self.num_heads,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0)
        self.talking_head2 = build_conv_layer(conv_cfg,
                                              self.num_heads,
                                              self.num_heads,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0)

        self.proj = ConvModule(in_channels=self.dh,
                               out_channels=dim,
                               kernel_size=1,
                               bias=True,
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg,
                               order=('act', 'conv', 'norm'))

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = ((q @ k) * self.scale + (self.attention_biases[:, self.attention_bias_idxs]
                                        if self.training else self.ab)
        )
        # attn = (q @ k) * self.scale
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v)

        out = x.transpose(2, 3).reshape(B, self.dh, self.resolution, self.resolution) + v_local
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out


class LGQuery(BaseModule):
    def __init__(self,
                 in_dim,
                 out_dim,
                 resolution1,
                 resolution2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(LGQuery,self).__init__(init_cfg=init_cfg)

        self.resolution1 = resolution1
        self.resolution2 = resolution2
        self.pool = nn.AvgPool2d(kernel_size=1, stride=2)

        self.local = ConvModule(in_channels=in_dim,
                                out_channels=out_dim,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_dim,
                                conv_cfg=conv_cfg,
                                norm_cfg=None,
                                act_cfg=None)

        self.proj = ConvModule(in_channels=in_dim,
                               out_channels=out_dim,
                               kernel_size=1,
                               padding=1,
                               bias=True,
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg,
                               act_cfg=None)

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention4DDownsample(BaseModule):
    def __init__(self,
                 dim=384,
                 out_dim=None,
                 resolution=7,
                 key_dim=16,
                 num_heads=8,
                 attn_ratio=4.0,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(Attention4DDownsample,self).__init__(init_cfg=init_cfg)

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        self.resolution = resolution

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = dim
        self.resolution2 = math.ceil(self.resolution / 2)
        self.q = LGQuery(dim, self.num_heads * self.key_dim, self.resolution, self.resolution2)

        self.N = self.resolution ** 2
        self.N2 = self.resolution2 ** 2

        self.k = ConvModule(in_channels=dim,
                            out_channels=out_dim,
                            kernel_size=1,
                            bias=True,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=None)

        self.v = ConvModule(in_channels=dim,
                            out_channels=self.num_heads * self.d,
                            kernel_size=1,
                            bias=True,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=None)

        self.v_local = ConvModule(in_channels=self.num_heads * self.d,
                                  out_channels=self.num_heads * self.d,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  groups=self.num_heads * self.d,
                                  bias=True,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=None)

        self.proj = ConvModule(in_channels=self.dh,
                               out_channels=self.out_dim,
                               kernel_size=1,
                               bias=True,
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg,
                               order=('act', 'conv', 'norm'))

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = ((q @ k) * self.scale + (self.attention_biases[:, self.attention_bias_idxs]
                                        if self.training else self.ab))

        # attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(2, 3)
        out = x.reshape(B, self.dh, self.resolution2, self.resolution2) + v_local

        out = self.proj(out)
        return out


class Embedding(BaseModule):
    def __init__(self, kernel_size=3, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_cfg=dict(type='BN'),
                 light=False, asub=False, resolution=None, conv_cfg=dict(type='Conv2d'),
                 act_cfg=dict(type='GELU'), init_cfg=None):
        super(Embedding, self).__init__(init_cfg=init_cfg)
        self.light = light
        self.asub = asub

        if self.light:
            self.new_proj = Sequential(
                ConvModule(in_channels=in_chans,
                           out_channels=in_chans,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           groups=in_chans,
                           bias=True,
                           conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg,
                           act_cfg=dict(type='HSwish')),
                ConvModule(in_channels=in_chans,
                           out_channels=embed_dim,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=True,
                           conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg,
                           act_cfg=None))

            self.skip = ConvModule(in_channels=in_chans,
                                   out_channels=embed_dim,
                                   kernel_size=1,
                                   stride=2,
                                   padding=0,
                                   bias=True,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=None)
        elif self.asub:
            self.attn = Attention4DDownsample(
                dim=in_chans,
                out_dim=embed_dim,
                resolution=resolution,
                act_cfg=act_cfg)

            self.conv = ConvModule(in_channels=in_chans,
                                   out_channels=embed_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=True,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=None)

        else:
            self.proj = ConvModule(in_channels=in_chans,
                                   out_channels=embed_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=True,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=None)

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out = self.attn(x) + out_conv
        else:
            x = self.proj(x)
            out = self.norm(x)
        return out


class ConvMlp(BaseModule):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 mid_conv=True,
                 drop_rate=0.,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(ConvMlp, self).__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv

        self.fc1 = ConvModule(in_channels=in_features,
                              out_channels=hidden_features,
                              kernel_size=1,
                              bias=True,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              act_cfg=act_cfg)

        self.drop = nn.Dropout(drop_rate)

        if self.mid_conv:
            self.mid = ConvModule(in_channels=hidden_features,
                                  out_channels=hidden_features,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  groups=hidden_features,
                                  bias=True,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg)

        self.fc2 = ConvModule(in_channels=hidden_features,
                              out_channels=out_features,
                              kernel_size=1,
                              bias=True,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              act_cfg=None)

    def forward(self, x):
        x = self.fc1(x)
        if self.mid_conv:
            x = self.mid(x)

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientFormerBlock(BaseModule):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 ues_attn=True,
                 resolution=7,
                 stride=None,
                 use_layer_scale=True,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer='Conv2d',
                         std=.02,
                         bias=0.),
                     dict(type='Constant', layer=['LayerScale'], val=1e-5)
                 ]):

        super(EfficientFormerBlock, self).__init__(init_cfg=init_cfg)
        self.use_attn = ues_attn
        if self.ues_attn:
            self.token_mixer = Attention4D(dim=dim,
                                           resolution=resolution,
                                           stride=stride,
                                           conv_cfg=conv_cfg,
                                           norm_cfg=norm_cfg,
                                           act_cfg=act_cfg)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(in_features=dim,
                           hidden_features=mlp_hidden_dim,
                           out_features=dim,
                           mid_conv=True,
                           drop_rate=drop,
                           conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.ls1 = LayerScale(dim, data_format='channels_first')
            self.ls2 = LayerScale(dim, data_format='channels_first')
        else:
            self.ls1, self.ls2 = nn.Identity(), nn.Identity()


    def forward(self, x):
        if self.use_attn:
            x = x + self.drop_path(self.ls1(self.token_mixer(x)))
        x = x + self.drop_path(self.ls2(self.mlp(x)))
        return x


def eformer_block(dim, index, layers, mlp_ratio=4.,
                  act_cfg=dict(type='GELU'), conv_cfg=dict(type='Conv2d'),
                  norm_cfg=dict(type='BN'),
                  drop_rate=.0, drop_path_rate=0.,
                  use_layer_scale=True, vit_num=1, resolution=7, e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            if index == 2:
                stride = 2
            else:
                stride = None
            blocks.append(EfficientFormerBlock(
                dim, mlp_ratio=mlp_ratio,
                ues_attn=True,
                act_cfg=act_cfg, norm_cfg=norm_cfg,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                conv_cfg=conv_cfg,
                resolution=resolution,
                stride=stride,
            ))
        else:
            blocks.append(EfficientFormerBlock(
                dim, mlp_ratio=mlp_ratio,
                ues_attn=False,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                conv_cfg=conv_cfg,
            ))
    blocks = nn.Sequential(*blocks)
    return blocks


@MODELS.register_module()
class EfficientFormerV2(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 pool_size=3,
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='GELU'),
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 vit_num=0,
                 distillation=True,
                 resolution=224,
                 e_ratios=expansion_ratios_L,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self._make_stem(3, embed_dims[0], norm_cfg=norm_cfg, act_cfg=act_cfg)

        network = []
        for i in range(len(layers)):
            stage = eformer_block(embed_dims[i], i, layers,
                                  pool_size=pool_size, mlp_ratio=mlp_ratios,
                                  act_layer=act_cfg, norm_layer=norm_cfg,
                                  drop_rate=drop_rate,
                                  drop_path_rate=drop_path_rate,
                                  use_layer_scale=use_layer_scale,
                                  layer_scale_init_value=layer_scale_init_value,
                                  resolution=math.ceil(resolution / (2 ** (i + 2))),
                                  vit_num=vit_num,
                                  e_ratios=e_ratios)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                if i >= 2:
                    asub = True
                else:
                    asub = False
                network.append(
                    Embedding(
                        kernel_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                        resolution=math.ceil(resolution / (2 ** (i + 2))),
                        asub=asub,
                        act_cfg=act_cfg, norm_cfg=norm_cfg,
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def _make_stem(self, in_channels, stem_channels, norm_cfg=dict(type='BN'), act_cfg=dict(type='GELU')):
        """make 2-ConvBNGELU stem layer."""
        self.patch_embed = Sequential(
            ConvModule(
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))


    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        # print(x.size())
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        # for image classification
        return cls_out


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


@register_model
def efficientformerv2_s0(pretrained=False, **kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S0'],
        embed_dims=EfficientFormer_width['S0'],
        downsamples=[True, True, True, True, True],
        vit_num=2,
        drop_path_rate=0.0,
        e_ratios=expansion_ratios_S0,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_model
def efficientformerv2_s1(pretrained=False, **kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S1'],
        embed_dims=EfficientFormer_width['S1'],
        downsamples=[True, True, True, True],
        vit_num=2,
        drop_path_rate=0.0,
        e_ratios=expansion_ratios_S1,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_model
def efficientformerv2_s2(pretrained=False, **kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S2'],
        embed_dims=EfficientFormer_width['S2'],
        downsamples=[True, True, True, True],
        vit_num=4,
        drop_path_rate=0.02,
        e_ratios=expansion_ratios_S2,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_model
def efficientformerv2_l(pretrained=False, **kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['L'],
        embed_dims=EfficientFormer_width['L'],
        downsamples=[True, True, True, True],
        vit_num=6,
        drop_path_rate=0.1,
        e_ratios=expansion_ratios_L,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model

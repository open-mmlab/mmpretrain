# Copyright (c) OpenMMLab. All rights reserved.x
import itertools
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

        self.N = self.resolution ** 2
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

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

        attn = ((q @ k) * self.scale +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab))
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
                                out_channels=in_dim,
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

    def forward(self, x):  # x (B,N,H,W)
        B, C, H, W = x.shape

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = (
                (q @ k) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )

        # attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(2, 3)
        out = x.reshape(B, self.dh, self.resolution2, self.resolution2) + v_local

        out = self.proj(out)
        return out


class Embedding(BaseModule):
    def __init__(self,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 in_chans=3,
                 embed_dim=768,
                 norm_cfg=dict(type='BN'),
                 light=False,
                 asub=False,
                 resolution=None,
                 conv_cfg=dict(type='Conv2d'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
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
            out = self.proj(x)
        return out


class ConvMlp(BaseModule):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]

    Compare with the ConvMLP in EfficientFormerV1, the module add
    a mid convolution layer.
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
        if self.use_attn:
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


@MODELS.register_module()
class EfficientFormerV2(BaseBackbone):

    # --layers: [x,x,x,x], numbers of layers for the four stages
    # --embed_dims: [x,x,x,x], embedding dims for the four stages
    # --downsamples: [x,x,x,x], has downsample or not in the four stages
    # --vit_num: (int), the num of vit blocks in the last stage
    # --expansion_ratios: [[],[],[],[]], the expansion ratio for each layer
    arch_settings = {
        's0': {
            'layers': [2, 2, 6, 4],
            'embed_dims': [32, 48, 96, 176],
            'vit_num': 2,
            'expansion_ratios': {
                '0': [4, 4],
                '1': [4, 4],
                '2': [4, 3, 3, 3, 4, 4],
                '3': [4, 3, 3, 4],
            }
        },
        's1': {
            'layers': [3, 3, 9, 6],
            'embed_dims': [32, 48, 120, 224],
            'vit_num': 2,
            'expansion_ratios': {
                '0': [4, 4, 4],
                '1': [4, 4, 4],
                '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
                '3': [4, 4, 3, 3, 4, 4],
            }
        },
        's2': {
            'layers': [4, 4, 12, 8],
            'embed_dims': [32, 64, 144, 288],
            'vit_num': 4,
            'expansion_ratios': {
                '0': [4, 4, 4, 4],
                '1': [4, 4, 4, 4],
                '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
                '3': [4, 4, 3, 3, 3, 3, 4, 4],
            }
        },
        'l': {
            'layers': [5, 5, 15, 10],
            'embed_dims': [40, 80, 192, 384],
            'vit_num': 6,
            'expansion_ratios': {
                '0': [4, 4, 4, 4, 4],
                '1': [4, 4, 4, 4, 4],
                '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
                '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
            }
        },
    }

    def __init__(self,
                 arch='s0',
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 resolution=224,
                 out_indices=(-1,),
                 frozen_stages=-1,
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
        super(EfficientFormerV2, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            default_keys = set(self.arch_settings['l1'].keys())
            assert set(arch.keys()) == default_keys, \
                f'The arch dict must have {default_keys}, ' \
                f'but got {list(arch.keys())}.'

        self.layers = arch['layers']  # [2, 2, 6, 4]
        self.embed_dims = arch['embed_dims']  # [32, 48, 96, 176]
        self.vit_num = arch['vit_num']
        self.expansion_ratios = arch['expansion_ratios']
        self.drop_path_rate = drop_path_rate

        assert isinstance(self.layers, list) and isinstance(
            self.embed_dims, list) and isinstance(
            self.vit_num, int), \
            f'layers and embed_dims should be List, vit_nums should be Int. ' \
            f'But got layer = {self.layers}, embed_dims = {self.embed_dims}, ' \
            f'vit_num = {self.vit_num}.'
        assert len(self.layers) == len(self.embed_dims), \
            f'The length of layers should be equal to embed_dims, ' \
            f'but got the length of layers = {len(self.layers)}, ' \
            f'the length of embed_dims = {len(self.embed_dims)}.'

        self._make_stem(in_channels=3,
                        stem_channels=self.embed_dims[0],
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)

        self.network = nn.ModuleList()

        for i, num_layer in enumerate(self.layers):
            blocks = []
            for idx in range(num_layer):
                block_dpr = self.drop_path_rate * (idx + sum(self.layers[:i])) / (sum(self.layers) - 1)
                if i >= 2 and idx > num_layer - 1 - self.vit_num:
                    use_attn = True
                    # stride represent whether to use stride_conv in attn layer in Block
                    stride = 2 if i == 2 else None
                else:
                    use_attn = False
                    stride = None
                blocks.append(EfficientFormerBlock(
                    self.embed_dims[i],
                    mlp_ratio=self.expansion_ratios[str(i)][idx],
                    ues_attn=use_attn,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    conv_cfg=conv_cfg,
                    resolution=math.ceil(resolution / (2 ** (i + 2))),
                    stride=stride))
            self.network.append(Sequential(*blocks))

            if i >= len(self.layers) - 1:
                break
            if self.embed_dims[i] != self.embed_dims[i + 1]:
                asub = True if i >= 2 else False
                self.network.append(
                    Embedding(
                        kernel_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=self.embed_dims[i],
                        embed_dim=self.embed_dims[i + 1],
                        resolution=math.ceil(resolution / (2 ** (i + 2))),
                        asub=asub,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 7 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        if len(self.out_indices) > 1:
            assert set(self.out_indices).issubset(set([0, 2, 4, 6])), \
                f'If use EfficientFormerV2 for Detection and Segmentation, ' \
                f'the output layer needs to be a subset of set(0,2,4,6), ' \
                f'but get {self.out_indices} '

        for i_layer in self.out_indices:
            layer = build_norm_layer(norm_cfg, self.embed_dims[i_layer // 2])[1]
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        self._freeze_stages()


    def _make_stem(self,
                   in_channels,
                   stem_channels,
                   conv_cfg=dict(type='Conv2d'),
                   norm_cfg=dict(type='BN'),
                   act_cfg=dict(type='GELU')):
        """make 2-ConvBNGELU stem layer."""
        self.patch_embed = Sequential(
            ConvModule(
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def _freeze_stages(self):
        if self.frozen_stages > 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            # Include both block and downsample layer.
            module = self.network[i]
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False


    def _format_output(self,x,idx):
        norm_layer = getattr(self, f'norm{idx}')
        return norm_layer(x)


    def forward(self, x):
        outs = []
        x = self.patch_embed(x)
        for idx, block in enumerate(self.network):
            x = block(x)
            if idx in self.out_indices:
                outs.append(self._format_output(x,idx))

        return tuple(outs)

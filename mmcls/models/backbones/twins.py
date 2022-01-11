# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import BaseModule, ModuleList
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.builder import BACKBONES
from mmcls.models.utils.attention import MultiheadAttention
from mmcls.models.utils.position_encoding import ConditionalPositionEncoding


class GlobalSubsampledAttention(MultiheadAttention):
    """Global Sub-sampled Attention (GSA) module.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 norm_cfg=dict(type='LN'),
                 qkv_bias=True,
                 sr_ratio=1,
                 **kwargs):
        super(GlobalSubsampledAttention,
              self).__init__(embed_dims, num_heads, **kwargs)

        self.qkv_bias = qkv_bias
        self.q = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.kv = nn.Linear(self.input_dims, embed_dims * 2, bias=qkv_bias)

        delattr(self,
                'qkv')  # remove self.qkv, here split into self.q, self.kv

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        assert H * W == N, 'The  product of h and w of hw_shape must be N, ' \
                           'which is the 2nd dim number of the input Tensor x.'

        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, *hw_shape)  # BNC_2_BCHW
            x = self.sr(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BCHW_2_BNC
            x = self.norm(x)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                self.head_dims).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class GSAEncoderLayer(BaseModule):
    """Implements one encoder layer with GSA.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (float): Kernel_size of conv in Attention modules. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1.,
                 init_cfg=None):
        super(GSAEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)[1]
        self.attn = GlobalSubsampledAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class LocallyGroupedSelfAttention(BaseModule):
    """Locally-grouped Self Attention (LSA) module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        window_size(int): Window size of LSA. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 window_size=1,
                 init_cfg=None):
        super(LocallyGroupedSelfAttention, self).__init__(init_cfg=init_cfg)

        assert embed_dims % num_heads == 0, f'dim {embed_dims} should be ' \
                                            f'divided by num_heads ' \
                                            f'{num_heads}.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.window_size = window_size

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of Local-groups
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # calculate attention mask for LSA
        Hp, Wp = x.shape[1:-1]
        _h, _w = Hp // self.window_size, Wp // self.window_size
        mask = torch.zeros((1, Hp, Wp), device=x.device)
        mask[:, -pad_b:, :].fill_(1)
        mask[:, :, -pad_r:].fill_(1)

        # [B, _h, _w, window_size, window_size, C]
        x = x.reshape(B, _h, self.window_size, _w, self.window_size,
                      C).transpose(2, 3)
        mask = mask.reshape(1, _h, self.window_size, _w,
                            self.window_size).transpose(2, 3).reshape(
                                1, _h * _w,
                                self.window_size * self.window_size)
        # [1, _h*_w, window_size*window_size, window_size*window_size]
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-1000.0)).masked_fill(
                                              attn_mask == 0, float(0.0))

        # [3, B, _w*_h, nhead, window_size*window_size, dim]
        qkv = self.qkv(x).reshape(B, _h * _w,
                                  self.window_size * self.window_size, 3,
                                  self.num_heads, C // self.num_heads).permute(
                                      3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [B, _h*_w, n_head, window_size*window_size, window_size*window_size]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask.unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.window_size,
                                                  self.window_size, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.window_size,
                                         _w * self.window_size, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LSAEncoderLayer(BaseModule):
    """Implements one encoder layer with LSA.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
           Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True
        qk_scale (float | None, optional): Override default qk scale of
           head_dim ** -0.5 if set. Default: None.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        window_size (int): Window size of LSA. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 qk_scale=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 window_size=1,
                 init_cfg=None):

        super(LSAEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)[1]
        self.attn = LocallyGroupedSelfAttention(embed_dims, num_heads,
                                                qkv_bias, qk_scale,
                                                attn_drop_rate, drop_rate,
                                                window_size)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


@BACKBONES.register_module()
class PCPVT(BaseModule):
    """The backbone of Twins-PCPVT.

    This backbone is the implementation of `Twins: Revisiting the Design
    of Spatial Attention in Vision Transformers
    <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (list): Embedding dimension. Default: [64, 128, 256, 512].
        patch_sizes (list): The patch sizes. Default: [4, 2, 2, 2].
        strides (list): The strides. Default: [4, 2, 2, 2].
        num_heads (int): Number of attention heads. Default: [1, 2, 4, 8].
        mlp_ratios (int): Ratio of mlp hidden dim to embedding dim.
            Default: [4, 4, 4, 4].
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool): Enable bias for qkv if True. Default: False.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): Stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        depths (list): Depths of each stage. Default [3, 4, 6, 3]
        sr_ratios (list): Kernel_size of conv in each Attn module in
            Transformer encoder layer. Default: [8, 4, 2, 1].
        norm_after_stage（bool): Add extra norm. Default False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 patch_sizes=[4, 2, 2, 2],
                 strides=[4, 2, 2, 2],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 norm_after_stage=False,
                 pretrained=None,
                 final_norm=True,
                 init_cfg=None):
        super(PCPVT, self).__init__(init_cfg=init_cfg)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.final_norm = final_norm

        # patch_embeds
        self.patch_embeds = ModuleList()
        self.position_encoding_drops = ModuleList()
        self.layers = ModuleList()

        for i in range(len(depths)):
            self.patch_embeds.append(
                PatchEmbed(
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dims=embed_dims[i],
                    conv_type='Conv2d',
                    kernel_size=patch_sizes[i],
                    stride=strides[i],
                    padding='corner',
                    norm_cfg=dict(type='LN')))

            self.position_encoding_drops.append(nn.Dropout(p=drop_rate))

        # PEGs in paper
        self.position_encodings = ModuleList([
            ConditionalPositionEncoding(embed_dim, embed_dim)
            for embed_dim in embed_dims
        ])

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for k in range(len(depths)):
            _block = ModuleList([
                GSAEncoderLayer(
                    embed_dims=embed_dims[k],
                    num_heads=num_heads[k],
                    feedforward_channels=mlp_ratios[k] * embed_dims[k],
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur + i],
                    num_fcs=2,
                    qkv_bias=qkv_bias,
                    act_cfg=dict(type='GELU'),
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[k]) for i in range(depths[k])
            ])
            self.layers.append(_block)
            cur += depths[k]

        # final_norm before GAP
        if final_norm:
            self.norm_name, norm = build_norm_layer(
                norm_cfg, embed_dims[-1], postfix=1)
            self.add_module(self.norm_name, norm)

        self.out_indices = out_indices
        self.norm_after_stage = norm_after_stage
        if self.norm_after_stage:
            self.norm_list = ModuleList()
            for dim in embed_dims:
                self.norm_list.append(build_norm_layer(norm_cfg, dim)[1])

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        if self.init_cfg is not None:
            super(PCPVT, self).init_weights()
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x):
        outputs = list()

        b = x.shape[0]

        for i in range(len(self.depths)):
            x, hw_shape = self.patch_embeds[i](x)
            h, w = hw_shape
            x = self.position_encoding_drops[i](x)
            for j, blk in enumerate(self.layers[i]):
                x = blk(x, hw_shape)
                if j == 0:
                    x = self.position_encodings[i](x, hw_shape)
            if self.norm_after_stage:
                x = self.norm_list[i](x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm(x)

            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

            if i in self.out_indices:
                outputs.append(x)

        return tuple(outputs)


@BACKBONES.register_module()
class SVT(PCPVT):
    """The backbone of Twins-SVT.

    This backbone is the implementation of `Twins: Revisiting the Design
    of Spatial Attention in Vision Transformers
    <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (list): Embedding dimension. Default: [64, 128, 256, 512].
        patch_sizes (list): The patch sizes. Default: [4, 2, 2, 2].
        strides (list): The strides. Default: [4, 2, 2, 2].
        num_heads (int): Number of attention heads. Default: [1, 2, 4].
        mlp_ratios (int): Ratio of mlp hidden dim to embedding dim.
            Default: [4, 4, 4].
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool): Enable bias for qkv if True. Default: False.
        drop_rate (float): Dropout rate. Default 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Default 0.0
        drop_path_rate (float): Stochastic depth rate. Default 0.2.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        depths (list): Depths of each stage. Default [4, 4, 4].
        sr_ratios (list): Kernel_size of conv in each Attn module in
            Transformer encoder layer. Default: [4, 2, 1].
        windiow_sizes (list): Window size of LSA. Default: [7, 7, 7],
        input_features_slice（bool): Input features need slice. Default: False.
        norm_after_stage（bool): Add extra norm. Default False.
        strides (list): Strides in patch-Embedding modules. Default: (2, 2, 2)
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256],
                 patch_sizes=[4, 2, 2, 2],
                 strides=[4, 2, 2, 2],
                 num_heads=[1, 2, 4],
                 mlp_ratios=[4, 4, 4],
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_cfg=dict(type='LN'),
                 depths=[4, 4, 4],
                 sr_ratios=[4, 2, 1],
                 windiow_sizes=[7, 7, 7],
                 norm_after_stage=True,
                 pretrained=None,
                 final_norm=True,
                 init_cfg=None):
        super(SVT,
              self).__init__(in_channels, embed_dims, patch_sizes, strides,
                             num_heads, mlp_ratios, out_indices, qkv_bias,
                             drop_rate, attn_drop_rate, drop_path_rate,
                             norm_cfg, depths, sr_ratios, norm_after_stage,
                             pretrained, final_norm, init_cfg)
        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        for k in range(len(depths)):
            for i in range(depths[k]):
                # Odd-numbered layers are GSA, even-numbered layers are LSA
                if i % 2 == 0:
                    self.layers[k][i] = \
                        LSAEncoderLayer(
                            embed_dims=embed_dims[k],
                            num_heads=num_heads[k],
                            feedforward_channels=mlp_ratios[k] * embed_dims[k],
                            drop_rate=drop_rate,
                            norm_cfg=norm_cfg,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=dpr[sum(depths[:k])+i],
                            qkv_bias=qkv_bias,
                            window_size=windiow_sizes[k])

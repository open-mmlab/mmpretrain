# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_norm_layer
from mmcv.cnn.bricks import ConvModule, DropPath
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.backbones.mlp_mixer import MixerBlock
from mmcls.models.utils import LePEAttention, resize_pos_embed, to_2tuple
from mmcls.models.utils.layer_scale import LayerScale
from mmcls.registry import MODELS


class LePEAttentionDWBlock(BaseModule):
    """LePEAttention Block with DW.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of heads.
        window_size (int): The window size of LePEAttention.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4..
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0..
        drop_path_rate (float): The drop path rate. Defaults to 0..
        ffn_cfg (dict): Configs of ffn. Defaults to an empty dict.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 ffn_cfg=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.with_cp = with_cp
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.split_size = window_size
        self.ffn_ratio = ffn_ratio
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=True)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.branch_num = 2
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(0.)

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.attns = nn.ModuleList([
            LePEAttention(
                embed_dims // 2,
                mode=i,
                split_size=self.split_size,
                num_heads=num_heads // 2,
                qk_scale=None,
                attn_drop=0.) for i in range(self.branch_num)
        ])

        _ffn_cfg = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': drop_rate,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path_rate),
            'act_cfg': dict(type='GELU'),
            **ffn_cfg
        }
        self.ffn = FFN(**_ffn_cfg)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.dw = Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=embed_dims)

    def forward(self, x, hw_shape):

        def _inner_forward(x, hw_shape):
            H, W = hw_shape
            B, L, C = x.shape
            assert L == H * W, 'flatten img_tokens has wrong size'
            img = self.norm1(x)
            qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1,
                                                             3).contiguous()

            x1 = self.attns[0](qkv[:, :, :, :C // 2], hw_shape)
            x2 = self.attns[1](qkv[:, :, :, C // 2:], hw_shape)
            attened_x = torch.cat([x1, x2], dim=2)
            attened_x = self.proj(attened_x)
            x = x + self.drop_path(attened_x)

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            B, L, C = x.shape
            x = x.permute(0, 2,
                          1).contiguous().reshape(B, C, hw_shape[0],
                                                  hw_shape[1])
            x = self.dw(x)
            x = x.reshape(B, C, L).permute(0, 2, 1).contiguous()
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, hw_shape)
        else:
            x = _inner_forward(x, hw_shape)
        return x


class MLPMixer(BaseModule):
    """Mlp-Mixer block.

    Args:
        num_tokens (int): The number of patched tokens.
        embed_dims (int): The feature dimension.
        token_expansion (float): The expansion ratio for tokens FFNs.
            Defaults to 0.5.
        channel_expansion (float): The expansion ratio for channels FFNs.
            Defaults to 4..
        depth (int): Number of successive mlp mixer blocks. Defaults to 1.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0..
        drop_path_rate (float): Stochastic depth rate. Defaults to 0..
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_tokens,
                 embed_dims,
                 token_expansion=0.5,
                 channel_expansion=4.,
                 depth=1,
                 drop_path_rate=0.,
                 drop_out=0.,
                 init_cfg=None):
        super(MLPMixer, self).__init__(init_cfg)
        layers = [
            MixerBlock(num_tokens, embed_dims,
                       int(embed_dims * token_expansion),
                       int(embed_dims * channel_expansion), drop_out,
                       drop_path_rate) for _ in range(depth)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FeatureGroupingAttention(BaseModule):
    """FeatureGroupingAttention for FeatureGroupingBlock.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 qkv_bias=False,
                 qk_scale=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.k_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, query, key, value):
        bq, nq, _ = query.shape
        bk, nk, _ = key.shape
        bv, nv, _ = value.shape

        q = query.reshape(bq, nq, self.num_heads,
                          self.head_dims).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(bk, nk, self.num_heads,
                                     self.head_dims).permute(0, 2, 1, 3)
        v = value.reshape(bv, nv, self.num_heads,
                          self.head_dims).permute(0, 2, 1, 3)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (bq, self.num_heads, nq, nk)

        out = (attn @ v).transpose(1, 2).reshape(bq, nq, self.embed_dims)

        return out


class FeatureUngroupingAttention(BaseModule):
    """Multi-head Attention Module for FeatureUngroupingBlock.

    The differences between FeatureUngroupingAttention & MultiheadAttention:
        1. Separate q_proj, k_proj and v_proj.

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
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        use_layer_scale (bool): Add a LayerScale. Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 use_layer_scale=False,
                 init_cfg=None):
        super(FeatureUngroupingAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.q_proj = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.k_proj = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.v_proj = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = build_dropout(dropout_layer)

        if use_layer_scale:
            self.gamma1 = LayerScale(embed_dims)
        else:
            self.gamma1 = nn.Identity()

    def forward(self, query, key, value):
        bq, nq, _ = query.shape
        bk, nk, _ = key.shape
        bv, nv, _ = value.shape

        q = self.q_proj(query).reshape(bq, nq, self.num_heads,
                                       self.head_dims).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(bk, nk, self.num_heads,
                                     self.head_dims).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(bv, nv, self.num_heads,
                                       self.head_dims).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bq, nq, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class FeatureUngroupingBlock(BaseModule):
    """Feature Ungrouping & Projection Block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of FeatureGroupingBlock heads.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4..
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to False.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` in FeatureGroupingBlock if set.
            Defaults to None.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0..
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0..
        drop_path_rate (float): The drop path rate. Defaults to 0..
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.with_cp = with_cp

        self.norm_x = build_norm_layer(norm_cfg, embed_dims)[1]
        self.norm_group_token = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = FeatureUngroupingAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        _ffn_cfg = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': drop_rate,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path_rate),
            'act_cfg': act_cfg,
        }
        self.ffn = FFN(**_ffn_cfg)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.proj = nn.Linear(embed_dims * 2, embed_dims, bias=True)

    def forward(self, x, group_token):

        def _inner_forward(x, group_token):
            query = x
            x = self.norm_x(x)
            group_token = self.norm_group_token(group_token)

            x = torch.cat(
                (query, self.drop_path(self.attn(x, group_token,
                                                 group_token))),
                dim=-1)
            x = self.proj(x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp:
            return cp.checkpoint(_inner_forward, x, group_token)
        else:
            return _inner_forward(x, group_token)


class FeatureGroupingBlock(BaseModule):
    """Feature Grouping Block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of FeatureGroupingBlock heads.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to False.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` in FeatureGroupingBlock if set.
            Defaults to None.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0..
        drop_path_rate (float): The drop path rate. Defaults to 0..
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.with_cp = with_cp

        self.norm_x = build_norm_layer(norm_cfg, embed_dims)[1]
        self.norm_group_token = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = FeatureGroupingAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate)

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, group_token):

        def _inner_forward(x, group_token):
            x = self.norm_x(x)
            group_token = self.norm_group_token(group_token)
            x = self.drop_path(self.attn(group_token, x, x))
            return x

        if self.with_cp:
            return cp.checkpoint(_inner_forward, x, group_token)
        else:
            return _inner_forward(x, group_token)


class GPBlock(BaseModule):
    """Group Propagation Block.

    Args:
        embed_dims (int): Number of input channels.
        depth (int): Number of successive mlp mixer blocks.
        num_group_heads (int): Number of FeatureGroupingBlock heads.
        num_ungroup_heads (int): Number of FeatureUngroupingBlock heads.
        num_group_token (int): The number of group tokens.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4..
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        group_qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` in FeatureGroupingBlock if set.
            Defaults to None.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0..
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0..
        drop_path_rate (float): The drop path rate. Defaults to 0..
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        group_att_cfg (Sequence[dict] | dict): The extra config of each
            FeatureGroupingBlock block. Defaults to empty dicts.
        mixer_cfg (Sequence[dict] | dict): The extra config of each
            MLPMixer block. Defaults to empty dicts.
        ungroup_att_cfg (Sequence[dict] | dict): The extra config of each
            FeatureUngroupingBlock block. Defaults to empty dicts.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 depth,
                 num_group_heads,
                 num_ungroup_heads,
                 num_group_token,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 group_qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cp=False,
                 group_att_cfg=dict(),
                 mixer_cfg=dict(),
                 ungroup_att_cfg=dict(),
                 init_cfg=None):

        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_group_token = num_group_token
        self.with_cp = with_cp

        self.group_token = nn.Parameter(
            torch.zeros(1, num_group_token, embed_dims))

        _group_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_group_heads,
            qkv_bias=qkv_bias,
            qk_scale=group_qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=0.,
            with_cp=with_cp)
        _group_att_cfg.update(group_att_cfg)
        self.group_layer = FeatureGroupingBlock(**_group_att_cfg)

        _mixer_cfg = dict(
            num_tokens=num_group_token,
            embed_dims=embed_dims,
            token_expansion=0.5,
            channel_expansion=4.0,
            depth=depth,
            drop_path_rate=drop_path_rate)
        _mixer_cfg.update(mixer_cfg)
        self.mixer = MLPMixer(**_mixer_cfg)

        _ungroup_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_ungroup_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            with_cp=with_cp)
        _ungroup_att_cfg.update(ungroup_att_cfg)
        self.un_group_layer = FeatureUngroupingBlock(**_ungroup_att_cfg)

        self.dwconv = ConvModule(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dims,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True))

    def init_weights(self):
        super(GPBlock, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.group_token, std=0.02)

    def forward(self, x, hw_shape):
        B, L, C = x.size()
        group_token = self.group_token.expand(x.size(0), -1, -1)
        gt = group_token

        gt = self.group_layer(x=x, group_token=gt)
        gt = self.mixer(gt)
        ungroup_tokens = self.un_group_layer(x=x, group_token=gt)
        ungroup_tokens = ungroup_tokens.permute(0, 2, 1).contiguous().reshape(
            B, C, hw_shape[0], hw_shape[1])
        proj_tokens = self.dwconv(ungroup_tokens).view(B, C, -1).permute(
            0, 2, 1).contiguous().view(B, L, C)
        return proj_tokens


class ConvPatchEmbed(PatchEmbed):
    """ConvPatchEmbed for GPViT.

    Args:
        in_channels (int): The num of input channels. Defaults to 3.
        embed_dims (int): The dimensions of embedding. Defaults to 768.
        num_convs (int): The number of convolutions. Defaults to 0.
        conv_type (str): The type of convolution
            to generate patch embedding. Defaults to "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Defaults to 16.
        stride (int): The slide stride of embedding conv.
            Defaults to 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Defaults to "corner".
        dilation (int): The dilation rate of embedding conv. Defaults to 1.
        bias (bool): Bias of embed conv. Defaults to True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Defaults to None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 num_convs=0,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super(ConvPatchEmbed, self).__init__(
            in_channels=64,
            embed_dims=embed_dims,
            conv_type=conv_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            norm_cfg=norm_cfg,
            input_size=input_size,
            init_cfg=init_cfg)

        self.stem = ConvModule(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True))

        if num_convs > 0:
            convs = []
            for _ in range(num_convs):
                convs.append(
                    ConvModule(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='ReLU', inplace=True)))
            self.convs = nn.Sequential(*convs)
        else:
            self.convs = nn.Identity()

        self.init_out_size = (self.init_out_size[0] // 2,
                              self.init_out_size[1] // 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.convs(x)

        return super().forward(x)


@MODELS.register_module()
class GPViT(BaseBackbone):
    """GPViT.

    A PyTorch implement of : `GPViT: A High Resolution Non-Hierarchical Vision
    Transformer with Group Propagation <https://arxiv.org/abs/2212.06795>`_

    Args:
        arch (str | dict): GPViT architecture. If use string,
            choose from 'L1', 'L2', 'L3', 'L4'. If use dict,
            it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **window_size** (int): The window size of LePEAttention.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **num_group_heads** (int): The number of heads in
              FeatureGroupingBlock.
            - **num_ungroup_heads** (int): The number of heads in
              FeatureUngroupingBlock.
            - **ffn_ratio** (float): The expansion ratio of feedforward network
              hidden layer channels.
            - **num_convs** (int): The number of convolutions of
              ConvPatchEmbed.
            - **mlpmixer_depth** (int): The number of blocks in mlp mixer.
            - **group_layers** (dict): The config for group layer. The key is
              index and the value is num_group_token.

            Defaults to 'L1'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0..
        drop_path_rate (float): stochastic depth rate. Defaults to 0..
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['L1', 'l1'], {
                'embed_dims': 216,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'num_convs': 0,
                'mlpmixer_depth': 1,
                'group_layers': {
                    1: 64,
                    4: 32,
                    7: 32,
                    10: 16
                },
            }),
        **dict.fromkeys(
            ['L2', 'l2'], {
                'embed_dims': 348,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'num_convs': 1,
                'mlpmixer_depth': 1,
                'group_layers': {
                    1: 64,
                    4: 32,
                    7: 32,
                    10: 16
                },
            }),
        **dict.fromkeys(
            ['L3', 'l3'], {
                'embed_dims': 432,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'num_convs': 1,
                'mlpmixer_depth': 1,
                'group_layers': {
                    1: 64,
                    4: 32,
                    7: 32,
                    10: 16
                },
            }),
        **dict.fromkeys(
            ['L4', 'l4'], {
                'embed_dims': 624,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'num_convs': 2,
                'mlpmixer_depth': 1,
                'group_layers': {
                    1: 64,
                    4: 32,
                    7: 32,
                    10: 16
                },
            }),
    }

    def __init__(self,
                 arch='L1',
                 img_size=224,
                 patch_size=8,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 with_cp=False,
                 init_cfg=None):
        super(GPViT, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'window_size', 'num_layers', 'num_heads',
                'num_group_heads', 'num_ungroup_heads', 'ffn_ratio',
                'num_convs', 'mlpmixer_depth', 'group_layers'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.window_size = self.arch_settings['window_size']
        self.img_size = to_2tuple(img_size)
        self.frozen_stages = frozen_stages

        # Set patch embedding
        assert patch_size % 2 == 0
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size // 2,
            stride=patch_size // 2,
            num_convs=self.arch_settings['num_convs'],
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = ConvPatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers

        for i in range(self.num_layers):
            _arch_settings = copy.deepcopy(self.arch_settings)
            if i not in _arch_settings['group_layers'].keys():
                _layer_cfg = dict(
                    embed_dims=self.embed_dims,
                    num_heads=_arch_settings['num_heads'],
                    window_size=self.window_size,
                    ffn_ratio=_arch_settings['ffn_ratio'],
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    norm_cfg=norm_cfg,
                    with_cp=with_cp)
                _layer_cfg.update(layer_cfgs[i])
                attn_layer = LePEAttentionDWBlock(**_layer_cfg)
                self.layers.append(attn_layer)
            else:
                _layer_cfg = dict(
                    embed_dims=self.embed_dims,
                    depth=_arch_settings['mlpmixer_depth'],
                    num_group_heads=_arch_settings['num_group_heads'],
                    num_ungroup_heads=_arch_settings['num_ungroup_heads'],
                    num_group_token=_arch_settings['group_layers'][i],
                    ffn_ratio=_arch_settings['ffn_ratio'],
                    drop_path_rate=dpr[i],
                    with_cp=with_cp)
                group_layer = GPBlock(**_layer_cfg)
                self.layers.append(group_layer)

        self.final_norm = final_norm
        if final_norm:
            self.add_module('lastnorm',
                            build_norm_layer(norm_cfg, self.embed_dims)[1])

        for i in out_indices:
            if i != self.num_layers - 1:
                self.add_module(f'norm{i}',
                                build_norm_layer(norm_cfg, self.embed_dims)[1])

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    def init_weights(self):
        super(GPViT, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            if i - 1 in self.out_indices and i - 1 != self.num_layers - 1:
                norm_layer = getattr(self, f'norm{i - 1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers) and self.final_norm:
            self.lastnorm.eval()
            for param in self.lastnorm.parameters():
                param.requires_grad = False

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1])))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode, 0)

    def forward(self, x):
        x, patch_resolution = self.patch_embed(x)
        assert (
            patch_resolution[0] % self.window_size == 0
            ) and (
                patch_resolution[1] % self.window_size == 0
                ),  \
            f'patch_resolution={patch_resolution} need to be dibided by'  \
            f' window_size={self.window_size}'
        pos_embed = resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0)

        x = x + pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)

            if i == len(self.layers) - 1:
                # final norm
                x = self.lastnorm(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm{i}')
                    patch_token = norm_layer(patch_token)
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs.append(patch_token)
        return tuple(outs)

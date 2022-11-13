# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, AdaptivePadding
from mmengine.model import BaseModule, ModuleList
from torch import nn

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

from mmcls.registry import MODELS
from ..utils import BEiTAttention, LayerScale, resize_pos_embed
from .vision_transformer import TransformerEncoderLayer, VisionTransformer


class ViTAdapterTransformerEncoderLayer(TransformerEncoderLayer):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, *args, window_size=0, **kwargs):
        super(ViTAdapterTransformerEncoderLayer,
              self).__init__(*args, **kwargs)

        self.window_size = window_size

    @staticmethod
    def window_reverse(windows, H, W, window_size):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    def forward(self, x, hw_shape):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            B, L, C = x.shape
            H, W = hw_shape
            assert L == H * W, \
                f"The query length {L} doesn't match the input " \
                f'shape ({H}, {W}).'

            x = x.view(B, H, W, C)

            window_size = self.window_size
            pad_r = (window_size - W % window_size) % window_size
            pad_b = (window_size - H % window_size) % window_size
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            H_pad, W_pad = x.shape[1], x.shape[2]

            # nW*B, window_size, window_size, C
            x = self.window_partition(x, window_size)
            # nW*B, window_size*window_size, C
            x = x.view(-1, window_size**2, C)

        x = self.attn(x)

        if self.window_size > 0:
            # merge windows
            x = x.view(-1, window_size, window_size, C)
            # B H' W' C
            x = self.window_reverse(x, H_pad, W_pad, window_size)

            if H != H_pad or W != W_pad:
                x = x[:, :H, :W, :].contiguous()

            x = x.view(B, H * W, C)

        x = shortcut + x
        x = self.ffn(self.norm2(x), identity=x)
        return x


class ViTAdapterBEiTTransformerEncoderLayer(ViTAdapterTransformerEncoderLayer):
    """Implements one encoder layer in BEiT.

    Comparing with conventional ``TransformerEncoderLayer``, this module
    adds weights to the shortcut connection. In addition, ``BEiTAttention``
    is used to replace the original ``MultiheadAttention`` in
    ``TransformerEncoderLayer``.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        layer_scale_init_value (float): The initialization value for
            the learnable scaling of attention and FFN.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        window_size (tuple[int]): The height and width of the window.
            Defaults to None.
        attn_drop_rate (float): The drop out rate for attention layer.
            Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        bias (bool | str): The option to add leanable bias for q, k, v. If bias
            is True, it will add leanable bias. If bias is 'qv_bias', it will
            only add leanable bias for q, v. If bias is False, it will not add
            bias for q, k, v. Default to 'qv_bias'.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='LN').
        attn_cfg (dict): The configuration for the attention layer.
            Defaults to an empty dict.
        ffn_cfg (dict): The configuration for the ffn layer.
            Defaults to ``dict(add_identity=False)``.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 layer_scale_init_value: float,
                 beit_window_size: Tuple[int, int],
                 window_size=0,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 bias: Union[str, bool] = 'qv_bias',
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 attn_cfg: dict = dict(),
                 ffn_cfg: dict = dict(add_identity=False),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        attn_cfg.update(dict(window_size=beit_window_size, qk_scale=None))

        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=0.,
            drop_rate=0.,
            num_fcs=num_fcs,
            qkv_bias=bias,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)

        self.window_size = window_size

        # overwrite the default attention layer in TransformerEncoderLayer
        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                bias=bias,
                is_cls_token=False))
        self.attn = BEiTAttention(**attn_cfg)

        # overwrite the default ffn layer in TransformerEncoderLayer
        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.ffn = FFN(**ffn_cfg)

        # NOTE: drop path for stochastic depth, we shall see if
        # this is better than dropout here
        dropout_layer = dict(type='DropPath', drop_prob=drop_path_rate)
        self.drop_path = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.gamma_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((embed_dims)),
            requires_grad=True)
        self.gamma_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((embed_dims)),
            requires_grad=True)

    def forward(self, x, hw_shape):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            B, L, C = x.shape
            H, W = hw_shape
            assert L == H * W,\
                f"The query length {L} doesn't match the input "\
                f'shape ({H}, {W}).'

            x = x.view(B, H, W, C)

            window_size = self.window_size
            pad_r = (window_size - W % window_size) % window_size
            pad_b = (window_size - H % window_size) % window_size
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            H_pad, W_pad = x.shape[1], x.shape[2]

            # nW*B, window_size, window_size, C
            x = self.window_partition(x, window_size)
            # nW*B, window_size*window_size, C
            x = x.view(-1, window_size**2, C)

        x = self.attn(x)

        if self.window_size > 0:
            # merge windows
            x = x.view(-1, window_size, window_size, C)
            # B H' W' C
            x = self.window_reverse(x, H_pad, W_pad, window_size)

            if H != H_pad or W != W_pad:
                x = x[:, :H, :W, :].contiguous()

            x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        return x


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for H_, W_ in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(
                0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    h, w = x.shape[2:]
    spatial_shapes = torch.as_tensor(
        [(int(math.ceil(h / 16) * i), int(math.ceil(w / 16) * i))
         for i in [2, 1, 0.5]],
        dtype=torch.long,
        device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points(
        [(math.ceil(h / 16), math.ceil(w / 16))], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(math.ceil(h / 16), math.ceil(w / 16))],
                                     dtype=torch.long,
                                     device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points(
        [(int(math.ceil(h // 16) * i), int(math.ceil(w / 16) * i))
         for i in [2, 1, 0.5]], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class VitAdapterFFN(BaseModule):
    """An implementation of VitAdapterFFN of VitAdapter.

    The differences between MixFFN & FFN:
        1. Introduce VitAdapterDWConv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg

        self.fc1 = nn.Linear(embed_dims, feedforward_channels)
        self.dwconv = VitAdapterDWConv(feedforward_channels)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(feedforward_channels, embed_dims)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x, hw_shape):
        x = self.fc1(x)
        x = self.dwconv(x, hw_shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VitAdapterDWConv(BaseModule):

    def __init__(self, dim=768, init_cfg=None):
        super().__init__(init_cfg)
        self.dwconv = Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=dim)

    def forward(self, x, hw_shape):
        B, _, C = x.shape
        split_position = [
            hw_shape[0] * 2 * hw_shape[1] * 2,
            hw_shape[0] * 2 * hw_shape[1] * 2 + hw_shape[0] * hw_shape[1]
        ]
        x1 = x[:, 0:split_position[0], :].transpose(1, 2).view(
            B, C, hw_shape[0] * 2, hw_shape[1] * 2).contiguous()
        x2 = x[:, split_position[0]:split_position[1], :].transpose(1, 2).view(
            B, C, *hw_shape).contiguous()
        x3 = x[:, split_position[1]:, :].transpose(1, 2).view(
            B, C, hw_shape[0] // 2, hw_shape[1] // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads=6,
                 num_points=4,
                 num_levels=1,
                 with_ffn=True,
                 ffn_ratio=0.25,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-06),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.query_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.feat_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
            batch_first=True)
        self.with_cp = with_cp
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        if with_ffn:
            self.ffn = VitAdapterFFN(
                embed_dims=embed_dims,
                feedforward_channels=mlp_hidden_dim,
                ffn_drop=drop_rate)
            self.ffn_norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.drop_path = DropPath(
                drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        else:
            self.ffn = nn.Identity()
            self.ffn_norm = nn.Identity()
            self.drop_path = nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index, hw_shape):

        def _inner_forward(query, feat):

            attn = self.attn(
                query=self.query_norm(query),
                value=self.feat_norm(feat),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=None)
            query = query + attn

            query = query + self.drop_path(
                self.ffn(self.ffn_norm(query), hw_shape))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads=6,
                 num_points=4,
                 num_levels=1,
                 norm_cfg=dict(type='LN', eps=1e-06),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.with_cp = with_cp
        self.query_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.feat_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
            batch_first=True)
        self.gamma = LayerScale(embed_dims)

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(
                query=self.query_norm(query),
                value=self.feat_norm(feat),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=None)
            return query + self.gamma(attn)

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads=6,
                 num_points=4,
                 norm_cfg=dict(type='LN', eps=1e-06),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 with_ffn=True,
                 ffn_ratio=0.25,
                 extra_extractor=False,
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.injector = Injector(
            embed_dims=embed_dims,
            num_levels=3,
            num_heads=num_heads,
            num_points=num_points,
            norm_cfg=norm_cfg,
            with_cp=with_cp)
        self.extractor = Extractor(
            embed_dims=embed_dims,
            num_levels=1,
            num_heads=num_heads,
            num_points=num_points,
            norm_cfg=norm_cfg,
            with_ffn=with_ffn,
            ffn_ratio=ffn_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_points=num_points,
                    norm_cfg=norm_cfg,
                    with_ffn=with_ffn,
                    ffn_ratio=ffn_ratio,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                    with_cp=with_cp) for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, hw_shape):
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2])
        for blk in blocks:
            x = blk(x, hw_shape)
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            hw_shape=hw_shape)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    hw_shape=hw_shape)
        return x, c


class SpatialPriorModule(BaseModule):

    def __init__(self,
                 patch_size,
                 inplanes=64,
                 embed_dim=384,
                 norm_cfg=dict(type='SyncBN'),
                 act_cfg=dict(type='ReLU'),
                 padding='corner',
                 init_cfg=None):
        super().__init__(init_cfg)

        self.adaptive_padding = AdaptivePadding(
            kernel_size=patch_size, stride=patch_size, padding=padding)

        self.stem = nn.Sequential(*[
            Conv2d(
                3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm_layer(norm_cfg, inplanes)[1],
            build_activation_layer(act_cfg),
            Conv2d(
                inplanes,
                inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, inplanes)[1],
            build_activation_layer(act_cfg),
            Conv2d(
                inplanes,
                inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, inplanes)[1],
            build_activation_layer(act_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            Conv2d(
                inplanes,
                2 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, 2 * inplanes)[1],
            build_activation_layer(act_cfg),
        ])
        self.conv3 = nn.Sequential(*[
            Conv2d(
                2 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, 4 * inplanes)[1],
            build_activation_layer(act_cfg),
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(
                4 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, 4 * inplanes)[1],
            build_activation_layer(act_cfg),
        ])
        self.fc1 = Conv2d(
            inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = Conv2d(
            2 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.fc3 = Conv2d(
            4 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.fc4 = Conv2d(
            4 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def forward(self, x):
        x = self.adaptive_padding(x)

        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4


@MODELS.register_module()
class VitAdapter(VisionTransformer):
    adapter_zoo = {
        **dict.fromkeys(
            ['s', 'small'],
            {
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
                'window_size': 14,
                # 2, 5, 8 11 for global attention
                'window_block_indexes': [0, 1, 3, 4, 6, 7, 9, 10]
            }),
        **dict.fromkeys(
            ['b', 'base'],
            {
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
                'window_size': 14,
                # 2, 5, 8 11 for global attention
                'window_block_indexes': [0, 1, 3, 4, 6, 7, 9, 10]
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'],
            {
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
                'window_size': 14,
                # 2, 5, 8 11 for global attention
                'window_block_indexes': [0, 1, 3, 4, 6, 7, 9, 10]
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'],
            {
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
                'window_size': 14,
                # 2, 5, 8 11 for global attention
                'window_block_indexes': [0, 1, 3, 4, 6, 7, 9, 10]
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'],
            {
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
                'window_size': 14,
                # 2, 5, 8 11 for global attention
                'window_block_indexes': [0, 1, 3, 4, 6, 7, 9, 10]
            }),
    }

    def __init__(self,
                 *args,
                 arch='base',
                 patch_size=16,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 spm_norm_cfg=dict(type='BN'),
                 spm_inplane=64,
                 deform_num_points=4,
                 deform_num_heads=6,
                 with_adapter_ffn=True,
                 add_vit_feature=True,
                 adapter_ffn_ratio=0.25,
                 use_extra_extractor=True,
                 out_indices=(0, 1, 2, 3),
                 adapter_settings=None,
                 with_cp=False,
                 **kwargs):

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.adapter_zoo), \
                f'Arch {arch} is not in default archs {set(self.adapter_zoo)}'
            self.adapter_settings = self.adapter_zoo[arch]
        else:
            essential_keys = {
                'interaction_indexes', 'window_size', 'window_block_indexes'
            }
            assert isinstance(
                adapter_settings, dict
                ) and essential_keys <= set(adapter_settings), \
                'Custom adapter_settings needs a dict with keys' \
                f'{essential_keys}'
            self.adapter_settings = adapter_settings

        self.window_size = self.adapter_settings['window_size']
        self.window_block_indexes = self.adapter_settings[
            'window_block_indexes']

        super().__init__(
            *args,
            arch=arch,
            patch_size=patch_size,
            with_cls_token=False,
            output_cls_token=False,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            out_indices=out_indices,
            final_norm=False,
            **kwargs)

        self.interaction_indexes = self.adapter_settings['interaction_indexes']
        self.add_vit_feature = add_vit_feature

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dims))
        self.spm = SpatialPriorModule(
            patch_size,
            inplanes=spm_inplane,
            embed_dim=self.embed_dims,
            norm_cfg=spm_norm_cfg)
        self.interactions = nn.Sequential(*[
            InteractionBlock(
                embed_dims=self.embed_dims,
                num_heads=deform_num_heads,
                num_points=deform_num_points,
                drop_path_rate=drop_path_rate,
                norm_cfg=norm_cfg,
                with_ffn=with_adapter_ffn,
                ffn_ratio=adapter_ffn_ratio,
                with_cp=with_cp,
                extra_extractor=((True if i == len(self.interaction_indexes) -
                                  1 else False) and use_extra_extractor))
            for i in range(len(self.interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 2, 2)
        self.norm1 = build_norm_layer(spm_norm_cfg, self.embed_dims)[1]
        self.norm2 = build_norm_layer(spm_norm_cfg, self.embed_dims)[1]
        self.norm3 = build_norm_layer(spm_norm_cfg, self.embed_dims)[1]
        self.norm4 = build_norm_layer(spm_norm_cfg, self.embed_dims)[1]

    def _build_layers(self, drop_rate, drop_path_rate, qkv_bias, norm_cfg,
                      beit_style, layer_scale_init_value, layer_cfgs):
        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                window_size=0
                if i not in self.window_block_indexes else self.window_size)
            _layer_cfg.update(layer_cfgs[i])
            if beit_style:
                _layer_cfg.update(
                    dict(
                        layer_scale_init_value=layer_scale_init_value,
                        beit_window_size=self.patch_resolution))
                _layer_cfg.pop('qkv_bias')
                self.layers.append(
                    ViTAdapterBEiTTransformerEncoderLayer(**_layer_cfg))
            else:
                self.layers.append(
                    ViTAdapterTransformerEncoderLayer(**_layer_cfg))

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)

        # add level embed
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            hw_shape,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, hw_shape)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(B, self.embed_dims, hw_shape[0] * 2,
                                     hw_shape[1] * 2).contiguous()
        c3 = c3.transpose(1, 2).view(B, self.embed_dims,
                                     *hw_shape).contiguous()
        c4 = c4.transpose(1, 2).view(B, self.embed_dims, hw_shape[0] // 2,
                                     hw_shape[1] // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(B, self.embed_dims,
                                        *hw_shape).contiguous()
            x1 = F.interpolate(
                x3,
                size=(c1.size(2), c1.size(3)),
                mode='bilinear',
                align_corners=False)
            x2 = F.interpolate(
                x3,
                size=(c2.size(2), c2.size(3)),
                mode='bilinear',
                align_corners=False)
            x4 = F.interpolate(
                x3,
                size=(c4.size(2), c4.size(3)),
                mode='bilinear',
                align_corners=False)
            c1 = c1 + x1
            c2 = c2 + x2
            c3 = c3 + x3
            c4 = c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return tuple([f1, f2, f3, f4])

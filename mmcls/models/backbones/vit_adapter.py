# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import (Conv2d, ConvModule, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import AdaptivePadding
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmengine.model import BaseModule, ModuleList
from torch import nn

from mmcls.registry import MODELS
from ..utils import LayerScale, resize_pos_embed
from .beit import BEiT, BEiTTransformerEncoderLayer
from .vision_transformer import TransformerEncoderLayer, VisionTransformer


def get_reference_points(spatial_shapes, device):
    """Create reference points for Injector's and Extractor's
    MultiScaleDeformableAttention."""
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
    """Create deform inputs for InteractionBlock."""
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
    """An implementation of VitAdapterFFN in VitAdapter.

    The differences between VitAdapterFFN & FFN:
        1. VitAdapterFFN introduces VitAdapterDWConv to encode positional
           information.

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
    """An implementation of VitAdapterDWConv in VitAdapter.

    The differences between VitAdapterDWConv & DWConv:
        1. Split multi stage features then apply DWConv.

    Args:
        dim (int): The feature dimension.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 dim=768,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.dwconv = Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
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
    """Multi Scale Feature Extractor in ViT-Adapter.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads. Defaults to 6.
        num_points (int): The number of sampling points for each query in each
            head of MultiScaleDeformableAttention. Defaults to 4.
        num_levels (int): The number of feature map used in
            Attention. Defaults to 1.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        with_ffn (bool): The option to use ffn. If True, it use ffn.
            Default to True.
        ffn_ratio (float): The number of expansion ratio of feedforward
            network hidden layer channels. Default to 0.25.
        value_proj_ratio (float): The expansion ratio of value_proj.
            Defaults to 1.0.
        use_extra_extractor (bool): The option to use extra Extractor in
            InteractionBlock. If True, it use extra Extractor.
            Default to False.
        norm_cfg (dict): Config dict for normalization.
            Defaults to ``dict(type='LN', eps=1e-06)``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=6,
                 num_points=4,
                 num_levels=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 with_ffn=True,
                 ffn_ratio=0.25,
                 value_proj_ratio=1.0,
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
            dropout=0.0,
            batch_first=True,
            value_proj_ratio=value_proj_ratio)
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
    """Spatial Feature Injector in ViT-Adapter.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads. Defaults to 6.
        num_points (int): The number of sampling points for each query in each
            head of MultiScaleDeformableAttention. Defaults to 4.
        num_levels (int): The number of feature map used in
            Attention. Defaults to 1.
        value_proj_ratio (float): The expansion ratio of value_proj.
            Defaults to 1.0.
        norm_cfg (dict): Config dict for normalization.
            Defaults to ``dict(type='LN', eps=1e-06)``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=6,
                 num_points=4,
                 num_levels=1,
                 value_proj_ratio=1.0,
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
            dropout=0.0,
            batch_first=True,
            value_proj_ratio=value_proj_ratio)
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
    """InteractionBlock in ViT-Adapter.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads. Defaults to 6.
        num_points (int): The number of sampling points for each query in each
            head of MultiScaleDeformableAttention. Defaults to 4.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        with_ffn (bool): The option to use ffn. If True, it use ffn.
            Default to True.
        ffn_ratio (float): The number of expansion ratio of feedforward
            network hidden layer channels. Default to 0.25.
        value_proj_ratio (float): The expansion ratio of value_proj.
            Defaults to 1.0.
        use_extra_extractor (bool): The option to use extra Extractor in
            InteractionBlock. If True, it use extra Extractor.
            Default to False.
        norm_cfg (dict): Config dict for normalization.
            Defaults to ``dict(type='LN', eps=1e-06)``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=6,
                 num_points=4,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 with_ffn=True,
                 ffn_ratio=0.25,
                 value_proj_ratio=1.0,
                 use_extra_extractor=False,
                 norm_cfg=dict(type='LN', eps=1e-06),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.injector = Injector(
            embed_dims=embed_dims,
            num_levels=3,
            num_heads=num_heads,
            num_points=num_points,
            value_proj_ratio=value_proj_ratio,
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
            value_proj_ratio=value_proj_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            with_cp=with_cp)
        if use_extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_points=num_points,
                    norm_cfg=norm_cfg,
                    with_ffn=with_ffn,
                    ffn_ratio=ffn_ratio,
                    value_proj_ratio=value_proj_ratio,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                    with_cp=with_cp) for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, hw_shape,
                **block_kwargs):
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2])
        for blk in blocks:
            x = blk(x, hw_shape=hw_shape, **block_kwargs)
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
    """SpatialPriorModule in ViT-Adapter.

    Args:
        patch_size (int | tuple): The patch size in patch embedding.
        embed_dims (int): The feature dimension.
        hidden_dims (int): Hidden dimension. Defaults to 64.
        norm_cfg (dict): Config dict for normalization.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): The activation config.
            Defaluts to ``dict(type='ReLU')``.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default to "corner".
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 patch_size,
                 embed_dims,
                 hidden_dims=64,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 padding='corner',
                 init_cfg=None):
        super().__init__(init_cfg)

        self.adaptive_padding = AdaptivePadding(
            kernel_size=patch_size, stride=patch_size, padding=padding)

        self.stem = nn.Sequential(*[
            ConvModule(
                in_channels,
                hidden_dims,
                3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                hidden_dims,
                hidden_dims,
                3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                hidden_dims,
                hidden_dims,
                3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = ConvModule(
            hidden_dims,
            2 * hidden_dims,
            3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            2 * hidden_dims,
            4 * hidden_dims,
            3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv4 = ConvModule(
            4 * hidden_dims,
            4 * hidden_dims,
            3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.fc1 = Conv2d(
            hidden_dims, embed_dims, kernel_size=1, stride=1, padding=0)
        self.fc2 = Conv2d(
            2 * hidden_dims, embed_dims, kernel_size=1, stride=1, padding=0)
        self.fc3 = Conv2d(
            4 * hidden_dims, embed_dims, kernel_size=1, stride=1, padding=0)
        self.fc4 = Conv2d(
            4 * hidden_dims, embed_dims, kernel_size=1, stride=1, padding=0)

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
    """ViT-Adapter.

    A PyTorch implement of : `Vision Transformer Adapter for Dense Predictions
    <https://arxiv.org/abs/2205.08534>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.
            - **interaction_indexes** (ListList[[int]]): The indexes of each
              interaction block.
            - **window_size** (int): The height and width of the window.
            - **window_block_indexes** (int): The indexes of window attention
              blocks.
            - **value_proj_ratio** (float): The expansion ratio of value_proj.

            Defaults to 'base'.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer of
            Vision Transformer. Defaults to ``dict(type='LN')``.
        spm_norm_cfg (dict): Config dict for normalization layer of Adapter.
            Defaults to ``dict(type='BN')``.
        spm_hidden_dims (int): Hidden dimension for SpatialPriorModule.
            Defaults to 64.
        deform_num_points (int): The number of sampling points for
            each query in each head of MultiScaleDeformableAttention.
            Default to 4.
        deform_num_heads (int): Parallel attention heads of
            MultiScaleDeformableAttention. Default to 64.
        with_adapter_ffn (bool): The option to use ffn for adapter. If True,
            it use ffn. Default to True.
        add_vit_feature (bool): The option to add vit feature to adapter
            feature. If True, it add vit feature. Default to True.
        adapter_ffn_ratio (float): The number of expansion ratio of feedforward
            network hidden layer channels of adapter. Default to 0.25.
        use_extra_extractor (bool): The option to use extra Extractor in
            InteractionBlock. If True, it use extra Extractor. Default to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """
    adapter_zoo = {
        **dict.fromkeys(
            ['b', 'base'],
            {
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
                'window_size': 14,
                # 2, 5, 8, 11 for global attention
                'window_block_indexes': [0, 1, 3, 4, 6, 7, 9, 10],
                'value_proj_ratio': 0.5
            }),
        **dict.fromkeys(
            ['l', 'large'],
            {
                'interaction_indexes': [[0, 5], [6, 11], [12, 17], [18, 23]],
                'window_size':
                14,
                # 5, 11, 17, 23 for global attention
                'window_block_indexes': [
                    0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19,
                    20, 21, 22
                ],
                'value_proj_ratio':
                0.5
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                'interaction_indexes': [[0, 7], [8, 15], [16, 23], [24, 31]],
                'window_size':
                14,
                # 7, 15, 23, 31 for global attention
                'window_block_indexes': [
                    0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18,
                    19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30
                ],
                'value_proj_ratio':
                0.5
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'],
            {
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
                'window_size': 14,
                # 2, 5, 8, 11 for global attention
                'window_block_indexes': [0, 1, 3, 4, 6, 7, 9, 10],
                'value_proj_ratio': 1.0
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'],
            {
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
                'window_size': 14,
                # 2, 5, 8, 11 for global attention
                'window_block_indexes': [0, 1, 3, 4, 6, 7, 9, 10],
                'value_proj_ratio': 1.0
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'],
            {
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
                'window_size': 14,
                # 2, 5, 8, 11 for global attention
                'window_block_indexes': [0, 1, 3, 4, 6, 7, 9, 10],
                'value_proj_ratio': 0.5
            }),
    }  # yapf: disable

    def __init__(self,
                 *args,
                 arch='base',
                 patch_size=16,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3),
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 spm_norm_cfg=dict(type='BN'),
                 spm_hidden_dims=64,
                 deform_num_points=4,
                 deform_num_heads=6,
                 with_adapter_ffn=True,
                 add_vit_feature=True,
                 adapter_ffn_ratio=0.25,
                 use_extra_extractor=True,
                 with_cp=False,
                 **kwargs):

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.adapter_zoo), \
                f'Arch {arch} is not in default archs {set(self.adapter_zoo)}'
            self.adapter_settings = self.adapter_zoo[arch]
        else:
            essential_keys = {
                'interaction_indexes', 'window_size', 'window_block_indexes',
                'value_proj_ratio'
            }
            assert isinstance(
                arch, dict
                ) and essential_keys <= set(arch), \
                'Custom adapter_settings needs a dict with keys' \
                f'{essential_keys}'
            self.adapter_settings = arch

        self.window_size = self.adapter_settings['window_size']
        self.window_block_indexes = self.adapter_settings[
            'window_block_indexes']
        self.value_proj_ratio = self.adapter_settings['value_proj_ratio']

        super().__init__(
            *args,
            arch=arch,
            patch_size=patch_size,
            in_channels=in_channels,
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
            in_channels,
            patch_size,
            self.embed_dims,
            hidden_dims=spm_hidden_dims,
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
                value_proj_ratio=self.value_proj_ratio,
                with_cp=with_cp,
                use_extra_extractor=(
                    (True if i == len(self.interaction_indexes) -
                     1 else False) and use_extra_extractor))
            for i in range(len(self.interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 2, 2)
        for i in out_indices:
            norm_layer = build_norm_layer(spm_norm_cfg, self.embed_dims)[1]
            self.add_module(f'norm{i}', norm_layer)

    def _build_layers(self, drop_rate, drop_path_rate, qkv_bias, norm_cfg,
                      layer_scale_init_value, layer_cfgs):
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
                layer_scale_init_value=layer_scale_init_value,
                window_size=0
                if i not in self.window_block_indexes else self.window_size)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

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
        outs = []
        for i, c in enumerate([c1, c2, c3, c4]):
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(c)
                outs.append(out)

        return tuple(outs)


@MODELS.register_module()
class BEiTAdapter(BEiT):
    """ViT-Adapter.

    A PyTorch implement of : `Vision Transformer Adapter for Dense Predictions
    <https://arxiv.org/abs/2205.08534>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.
            - **interaction_indexes** (List[List[int]]): The indexes of each
              interaction block.
            - **window_size** (List[int]): The height and width of the window.
            - **value_proj_ratio** (float): The expansion ratio of value_proj.

            Defaults to 'base'.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer of
            Vision Transformer. Defaults to ``dict(type='LN')``.
        spm_norm_cfg (dict): Config dict for normalization layer of Adapter.
            Defaults to ``dict(type='BN')``.
        spm_hidden_dims (int): Hidden dimension for SpatialPriorModule.
            Defaults to 64.
        deform_num_points (int): The number of sampling points for
            each query in each head of MultiScaleDeformableAttention.
            Default to 4.
        deform_num_heads (int): Parallel attention heads of
            MultiScaleDeformableAttention. Default to 64.
        with_adapter_ffn (bool): The option to use ffn for adapter. If True,
            it use ffn. Default to True.
        add_vit_feature (bool): The option to add vit feature to adapter
            feature. If True, it add vit feature. Default to True.
        adapter_ffn_ratio (float): The number of expansion ratio of feedforward
            network hidden layer channels of adapter. Default to 0.25.
        use_extra_extractor (bool): The option to use extra Extractor in
            InteractionBlock. If True, it use extra Extractor. Default to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """
    adapter_zoo = {
        **dict.fromkeys(
            ['b', 'base'],
            {
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
                'window_size': [14, 14, 14, 14, 14, 56,
                                14, 14, 14, 14, 14, 56,
                                14, 14, 14, 14, 14, 56,
                                14, 14, 14, 14, 14, 56],
                'value_proj_ratio': 0.5
            }),
    }  # yapf: disable

    def __init__(self,
                 *args,
                 arch='base',
                 patch_size=16,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3),
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 spm_norm_cfg=dict(type='BN'),
                 spm_hidden_dims=64,
                 deform_num_points=4,
                 deform_num_heads=6,
                 with_adapter_ffn=True,
                 add_vit_feature=True,
                 adapter_ffn_ratio=0.25,
                 use_extra_extractor=True,
                 with_cp=False,
                 **kwargs):

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.adapter_zoo), \
                f'Arch {arch} is not in default archs {set(self.adapter_zoo)}'
            self.adapter_settings = self.adapter_zoo[arch]
        else:
            essential_keys = {
                'interaction_indexes', 'window_size', 'value_proj_ratio'
            }
            assert isinstance(
                arch, dict
                ) and essential_keys <= set(arch), \
                'Custom adapter_settings needs a dict with keys' \
                f'{essential_keys}'
            self.adapter_settings = arch

        self.window_size = self.adapter_settings['window_size']
        self.value_proj_ratio = self.adapter_settings['value_proj_ratio']

        super().__init__(
            *args,
            arch=arch,
            patch_size=patch_size,
            in_channels=in_channels,
            with_cls_token=False,
            output_cls_token=False,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            out_indices=out_indices,
            final_norm=False,
            avg_token=False,
            **kwargs)

        self.interaction_indexes = self.adapter_settings['interaction_indexes']
        self.add_vit_feature = add_vit_feature

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dims))
        self.spm = SpatialPriorModule(
            in_channels,
            patch_size,
            self.embed_dims,
            hidden_dims=spm_hidden_dims,
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
                value_proj_ratio=self.value_proj_ratio,
                with_cp=with_cp,
                use_extra_extractor=(
                    (True if i == len(self.interaction_indexes) -
                     1 else False) and use_extra_extractor))
            for i in range(len(self.interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 2, 2)
        for i in out_indices:
            norm_layer = build_norm_layer(spm_norm_cfg, self.embed_dims)[1]
            self.add_module(f'norm{i}', norm_layer)

    def _build_layers(
        self,
        drop_rate,
        drop_path_rate,
        norm_cfg,
        layer_scale_init_value,
        layer_cfgs,
        use_rel_pos_bias,
    ):
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
                layer_scale_init_value=layer_scale_init_value,
                use_rel_pos_bias=use_rel_pos_bias,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg,
                use_window_attention=True,
                window_size=(self.window_size[i], self.window_size[i]),
                with_cls_token=self.with_cls_token)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(BEiTTransformerEncoderLayer(**_layer_cfg))

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

        if self.pos_embed is not None:
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                hw_shape,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        rel_pos_bias = self.rel_pos_bias() \
            if self.rel_pos_bias is not None else None

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.layers[indexes[0]:indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                hw_shape,
                rel_pos_bias=rel_pos_bias)

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
        outs = []
        for i, c in enumerate([c1, c2, c3, c4]):
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(c)
                outs.append(out)

        return tuple(outs)

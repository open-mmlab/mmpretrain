# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Callable, Tuple, Optional, Sequence, List, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.bricks import ConvModule, DropPath, build_activation_layer, build_conv_layer
from mmengine.model import BaseModule, Sequential
from mmengine.model.weight_init import trunc_normal_


from .base_backbone import BaseBackbone
from mmcls.models.utils import SELayer, make_divisible
from mmcls.registry import MODELS
from ..utils import build_norm_layer, to_2tuple, LayerNorm2d


#### TODO 1 : the GELU active function in pytorch 1.12 and 1.13 version add a new
####          parameter ————  approximate='none' , but mmcls and old version in pytorch
####          isnot support this parameter. Details in 
####          https://pytorch.org/docs/1.13/generated/torch.nn.GELU.html?highlight=nn+gelu#torch.nn.GELU


class MBConv(BaseModule):
    """ MBConv layer in MaxViT Backbone.

    A PyTorch implementation of MBConv introduced by:
    `MaxViT: Multi-Axis Vision Transformer
    <https://arxiv.org/pdf/2204.01697.pdf>`_

    Note: This implementation differs slightly from the original MobileNet implementation!
          This implementation differs slightly from the original EfficientnetV2 implementation!

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of output channels.
        stride (int) : Stride of the convolution layer.
            Defaults to 1.
        expand_ratio (float) : Expend ratio in mid_channels, the mid_channel will
            be ``make_divisible(channels * ratio, divisor)``. Defaults to 4.0
        drop_path (float): The ratio of the stochastic depth.
            Defaults to 0.0.
        conv_cfg (Tuple[dict]): Config dict for convolution layer.
            Defaults to (dict(type='Conv2d'),dict(type='Conv2dAdaptivePadding')),
            which means using two different conv modules.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', eps=1e-3).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='GELU').
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 expand_ratio: float = 4.0,
                 drop_path: float = 0.,
                 conv_cfg=(dict(type='Conv2d'),dict(type='Conv2dAdaptivePadding')),
                 norm_cfg=dict(type='BN', eps=1e-3),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(MBConv, self).__init__(init_cfg=init_cfg)

        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        if stride == 1:
            assert in_channels == out_channels, \
                "If stride is 1, input and output channels must be equal."

        if stride == 2 and in_channels != out_channels:
            self.shortcut = Sequential(
                nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
                build_conv_layer(
                    conv_cfg[0],
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                ))
        elif stride == 2 and in_channels == out_channels:
            self.shortcut = Sequential(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))
        else:
            self.shortcut = nn.Identity()

        mid_channels = make_divisible(out_channels * expand_ratio, 8)

        self.pre_norm = build_norm_layer(norm_cfg, in_channels)
        self.layers = nn.ModuleList()
        self.layers.append(ConvModule(in_channels=in_channels,
                                      out_channels=mid_channels,
                                      kernel_size=1,
                                      stride=1,
                                      conv_cfg=conv_cfg[0],
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg))

        self.layers.append(ConvModule(in_channels=mid_channels,
                                      out_channels=mid_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      groups=mid_channels,
                                      conv_cfg=conv_cfg[1],
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg))

        self.layers.append(SELayer(channels=mid_channels,
                                   ratio=16,
                                   conv_cfg=None,
                                   act_cfg=(dict(type='SiLU'), dict(type='Sigmoid'))))

        self.layers.append(ConvModule(in_channels=mid_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      conv_cfg=conv_cfg[0],
                                      norm_cfg=None,
                                      act_cfg=None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        Args:
            input (torch.Tensor): Input tensor of the shape
            [B, C_in, H, W].
        Returns:
            output (torch.Tensor): Output tensor of the shape
            [B, C_out, H (// 2), W (// 2)].
        """
        shortcut = self.shortcut(x)
        x = self.pre_norm(x)
        for i,layer in enumerate(self.layers):
            x = layer(x)
        x = self.drop_path(x) + shortcut
        return x


def reindex_2d_einsum_lookup(
        relative_position_tensor,
        height: int,
        width: int,
        height_lookup: torch.Tensor,
        width_lookup: torch.Tensor) -> torch.Tensor:
    """Reindex 2d relative position bias with 2 independent einsum lookups.

    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py # noqa: E501

    Args:
        relative_position_tensor: Tensor of shape
            [..., vocab_height, vocab_width, ...].
        height: height to reindex to.
        width: width to reindex to.
        height_lookup: one-hot height lookup
        width_lookup: one-hot width lookup
    Returns:
        reindexed_tensor: a Tensor of shape
            [..., height * width, height * width, ...]
    """
    reindexed_tensor = torch.einsum('nhw,ixh->nixw', relative_position_tensor, height_lookup)
    reindexed_tensor = torch.einsum('nixw,jyw->nijxy', reindexed_tensor, width_lookup)
    area = height * width
    return reindexed_tensor.reshape(relative_position_tensor.shape[0], area, area)


def generate_lookup_tensor(
        length: int,
        max_relative_position: Optional[int] = None):
    """Generate a one_hot lookup tensor to reindex embeddings along one dimension.

    Args:
        length: the length to reindex to.
        max_relative_position: the maximum relative position to consider.
            Relative position embeddings for distances above this threshold
            are zeroed out.
    Returns:
        a lookup Tensor of size [length, length, vocab_size] that satisfies
            ret[n,m,v] = 1{m - n + max_relative_position = v}.
    """
    if max_relative_position is None:
        max_relative_position = length - 1

    vocab_size = 2 * max_relative_position + 1
    ret = torch.zeros(length, length, vocab_size)
    for i in range(length):
        for x in range(length):
            v = x - i + max_relative_position
            if abs(x - i) > max_relative_position:
                continue
            ret[i, x, v] = 1
    return ret


class RelPosBiasTf(BaseModule):
    """ Relative Position Bias Impl (Compatible with Tensorflow MaxViT models)
    Adapted from:
     https://github.com/google-research/maxvit/blob/main/maxvit/models/attention_utils.py # noqa: E501

     Args:
         window_size (tuple[int]): Grid/Window size to be utilized.
         num_heads (int): Number of attention heads.
         prefix_tokens (int): Useless parameter.
            Defaults to 0.
         init_cfg (dict | list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 window_size: Tuple[int],
                 num_heads: int,
                 prefix_tokens: int = 0,
                 init_cfg=None):
        super(RelPosBiasTf,self).__init__(init_cfg=init_cfg)

        assert prefix_tokens <= 1, f'The prefix_tokens in RelPosBiasTf ' \
                                   f'module should <= 1, but get {RelPosBiasTf}'
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.num_heads = num_heads

        vocab_height = 2 * window_size[0] - 1
        vocab_width = 2 * window_size[1] - 1
        self.bias_shape = (self.num_heads, vocab_height, vocab_width)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.bias_shape))
        self.register_buffer('height_lookup', generate_lookup_tensor(window_size[0]), persistent=False)
        self.register_buffer('width_lookup', generate_lookup_tensor(window_size[1]), persistent=False)
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def get_bias(self) -> torch.Tensor:
        return reindex_2d_einsum_lookup(
            self.relative_position_bias_table,
            self.window_size[0],
            self.window_size[1],
            self.height_lookup,
            self.width_lookup
        )

    def forward(self, attn, shared_rel_pos = None):
        return attn + self.get_bias()


class AttentionCl(BaseModule):
    """ Channels-last multi-head attention (B, ..., C) layer in MaxViT Backbone.

    A PyTorch implementation of Channels-last multi-head attention introduced by:
    `MaxViT: Multi-Axis Vision Transformer
    <https://arxiv.org/pdf/2204.01697.pdf>`_

    Args:
        dim (int): Number of input channels.
        dim_out (Optional[int]): The output dimensions.
            Defaults to None, means dim_out = dim.
        dim_head (int): In order to compute the num_heads and scales.
            Defaults to 32.
        bias (bool): The bias in the Linear layer.
            Defaults to True.
        expand_first (bool): Whether to expend the dimensions
            Defaults to True.
        head_first (bool): Whether to put ``num_heads`` first.
            Defaults to False.
        rel_pos_cls (Callable): Relative Position Bias module.
            Defaults to None,
        attn_drop (float): The drop out rate for attention output weights.
            Defaults to 0.
        proj_drop (float): The drop out rate for Linear output weights.
            Defaults to 0.
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """


    def __init__(self,
                 dim: int,
                 dim_out: Optional[int] = None,
                 dim_head: int = 32,
                 bias: bool = True,
                 expand_first: bool = True,
                 head_first: bool = False,
                 rel_pos_cls: Callable = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 init_cfg=None):
        super(AttentionCl,self).__init__(init_cfg=init_cfg)

        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first and dim_out > dim else dim
        assert dim_attn % dim_head == 0, 'attn dim should be divisible by head_dim'
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5
        self.fast_attn = hasattr(F, 'scaled_dot_product_attention')  # FIXME

        self.qkv = nn.Linear(dim, dim_attn * 3, bias=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attn, dim_out, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        if self.head_first:
            q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        else:
            q, k, v = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)

        if self.fast_attn:
            if self.rel_pos is not None:
                attn_bias = self.rel_pos.get_bias()
            elif shared_rel_pos is not None:
                attn_bias = shared_rel_pos
            else:
                attn_bias = None
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop_rate,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if self.rel_pos is not None:
                attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
            elif shared_rel_pos is not None:
                attn = attn + shared_rel_pos
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size: Tuple[int]) -> torch.Tensor:
    """ Window partition function.

    Args:
        x (torch.Tensor): Input tensor of the shape [B, C, H, W].
        window_size (Tuple[int]): Window size to be applied.
    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape
            [B * windows, window_size[0], window_size[1], C].
    """

    B, H, W, C = x.shape
    assert H % window_size[0] == 0, \
        f'height ({H}) must be divisible by window ({window_size[0]})'
    assert W % window_size[1] == 0, \
        f'height ({W}) must be divisible by window ({window_size[1]})'
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size: Tuple[int], img_size: Tuple[int]):
    """ Reverses the window partition.

    Args:
        windows (torch.Tensor): Window tensor of the shape
            [B * windows, window_size[0], window_size[1], C].
        window_size (Tuple[int]): Window size which have been applied.
        img_size (Tuple[int]): Image shape.
    Returns:
        output (torch.Tensor): Folded output tensor of the shape
            [B, C, original_size[0], original_size[1]].
    """

    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition(x, grid_size: Tuple[int]):
    """ Grid partition function.

    Args:
        x (torch.Tensor): Input tensor of the shape [B, C, H, W].
        grid_size (Tuple[int]): Grid size to be applied.
    Returns:
        grid (torch.Tensor): Unfolded input tensor of the shape
            [B * grids, grid_size[0], grid_size[1], C].
    """

    B, H, W, C = x.shape
    assert H % grid_size[0] == 0, \
        f'height {H} must be divisible by grid {grid_size[0]}'
    assert W % grid_size[1] == 0, \
        f'height {W} must be divisible by grid {grid_size[1]}'
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


def grid_reverse(windows, grid_size: Tuple[int], img_size: Tuple[int]):
    """ Reverses the grid partition.
    Args:
        windows (torch.Tensor): Grid tensor of the shape
            [B * grids, grid_size[0], grid_size[1], C].
        grid_size (Tuple[int]): Grid size which have been applied.
        img_size (Tuple[int]): Original shape.
    Returns:
        output (torch.Tensor): Folded output tensor of the shape
            [B, C, original_size[0], original_size[1]].
    """

    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


class PartitionAttentionCl(BaseModule):
    """ Grid or Block partition Attention + FFN layer.
    NxC 'channels last' tensor layout.

    A PyTorch implementation of
    Grid or Block partition Attention + FFN layer
    introduced by:
    `MaxViT: Multi-Axis Vision Transformer
    <https://arxiv.org/pdf/2204.01697.pdf>`_

    Args:
        dim (int): Number of input channels.
        partition_type (str): Use Grid partition Attention or Block partition Attention.
            Defaults to 'block'.
        window_size (Tuple[int]): Window size to be applied.
            Defaults to (7,7).
        grid_size (Tuple[int]): Grid size to be applied.
            Defaults to (7,7).
        dim_head (int): In order to compute the num_heads and scales in
            Channel-last Attention layer. Defaults to 32.
        attn_bias (bool): The bias in the Linear layer in Channel-last Attention layer.
            Defaults to True.
        head_first (bool): Whether to put ``num_heads`` first in
            Channel-last Attention layer. Defaults to False,
        attn_drop (float): The drop out rate for attention output weights in
            Channel-last Attention layer. Defaults to 0.
        proj_drop (float): The drop out rate for Linear output weights in
            Channel-last Attention layer. Defaults to 0.
        ffn_expand_ratio (float) : Expend ratio in FFN layer, the mid_channel will
            be ``dim * ffn_expand_ratio``. Defaults to 4.0
        ffn_drop (float): The drop out rate for FFN output weights.
            Defaults to 0.
        drop_path (float): The ratio of the stochastic depth.
            Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='LN', eps=1e-5),
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='GELU').
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 dim: int,
                 partition_type: str = 'block',
                 window_size: Tuple[int] = (7, 7),
                 grid_size: Tuple[int] = (7, 7),
                 dim_head: int = 32,
                 attn_bias: bool = True,
                 head_first: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 ffn_expand_ratio: float = 4.0,
                 ffn_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_cfg=dict(type='LN', eps=1e-5),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(PartitionAttentionCl, self).__init__(init_cfg=init_cfg)

        assert partition_type in {None, 'block', 'grid'}
        self.partition_block = partition_type == 'block'
        self.partition_size = to_2tuple(window_size if self.partition_block else grid_size)
        rel_pos_cls = partial(RelPosBiasTf, window_size=window_size)

        self.norm1 = build_norm_layer(norm_cfg, dim)
        self.attn = AttentionCl(dim=dim,
                                dim_out=dim,
                                dim_head=dim_head,
                                bias=attn_bias,
                                head_first=head_first,
                                rel_pos_cls=rel_pos_cls,
                                attn_drop=attn_drop,
                                proj_drop=proj_drop,
        )

        # self.ls1 and self.ls2 can be changed by LayerScale
        self.ls1 = nn.Identity()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)
        self.ffn = FFN(embed_dims=dim,
                       feedforward_channels=int(dim * ffn_expand_ratio),
                       act_cfg=act_cfg,
                       ffn_drop=ffn_drop)

        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[1:3]
        if self.partition_block:
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)

        partitioned = self.attn(partitioned)

        if self.partition_block:
            x = window_reverse(partitioned, self.partition_size, img_size)
        else:
            x = grid_reverse(partitioned, self.partition_size, img_size)
        return x

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self._partition_attn(x)
        x = self.ls1(x)
        x = shortcut + self.drop_path1(x)

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x,identity=0)
        x = self.ls2(x)
        x = shortcut + self.drop_path2(x)
        return x


class MaxViTBlock(BaseModule):
    """ A MaxViT layer including:
    MaxVit conv, Window partition Attention + FFN, Grid partition Attention + FFN

    A PyTorch implementation of MaxViT layer introduced by:
    `MaxViT: Multi-Axis Vision Transformer
    <https://arxiv.org/pdf/2204.01697.pdf>`_

    Args:
        dim (int): Number of input channels.
        dim_out (int): Number of output channels in MBConv layer.
        stride (int) : Stride of the convolution layer in MBConv layer.
            Defaults to 1.
        expand_ratio_conv (float) : Expend ratio in mid_channels in MBConv layer.
            The mid_channel will be ``make_divisible(channels * ratio, divisor)``.
            Defaults to 4.0
        conv_cfg (Tuple[dict]): Config dict for convolution layer in MBConv layer.
            Defaults to (dict(type='Conv2d'),dict(type='Conv2dAdaptivePadding')),
            which means using two different conv modules.
        norm_cfg_conv (dict): Config dict for normalization layer in MBConv layer.
            Defaults to dict(type='BN', eps=1e-3).
        act_cfg_conv (dict): Config dict for activation layer in MBConv layer.
            Defaults to dict(type='GELU').
        window_size (Tuple[int]): Window size to be applied in
            Channel-last Attention layers. Defaults to (7,7).
        grid_size (Tuple[int]): Grid size to be applied in Channel-last Attention layers.
            Defaults to (7,7).
        dim_head (int): In order to compute the num_heads and scales in
            Channel-last Attention layers. Defaults to 32.
        attn_bias (bool): The bias in the Linear layer in Channel-last Attention layer.
            Defaults to True.
        head_first (bool): Whether to put ``num_heads`` first in
            Channel-last Attention layer. Defaults to False,
        attn_drop (float): The drop out rate for attention output weights in
            Channel-last Attention layer. Defaults to 0.
        proj_drop (float): The drop out rate for Linear output weights in
            Channel-last Attention layer. Defaults to 0.
        ffn_expand_ratio (float) : Expend ratio in FFN layer in Channel-last Attention layer,
            the mid_channel will be ``dim * ffn_expand_ratio``. Defaults to 4.0
        ffn_drop (float): The drop out rate for FFN output weights in
            Channel-last Attention layer. Defaults to 0.
        drop_path (float): The ratio of the stochastic depth.
            Defaults to 0.
        norm_cfg_transformer (dict): Config dict for normalization layer in
            Channel-last Attention layer. Defaults to dict(type='LN', eps=1e-5).
        act_cfg_transformer (dict): Config dict for activation layer in
            Channel-last Attention layer. Defaults to dict(type='GELU').
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 dim: int,
                 dim_out: int,
                 stride: int = 1,
                 expand_ratio_conv: float = 4.0,
                 conv_cfg=(dict(type='Conv2d'), dict(type='Conv2dAdaptivePadding')),
                 norm_cfg_conv=dict(type='BN', eps=1e-3),
                 act_cfg_conv=dict(type='GELU'),
                 window_size: Tuple[int] = (7, 7),
                 grid_size: Tuple[int] = (7, 7),
                 dim_head: int = 32,
                 attn_bias: bool = True,
                 head_first: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 ffn_expand_ratio: float = 4.,
                 ffn_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_cfg_transformer=dict(type='LN', eps=1e-5),
                 act_cfg_transformer=dict(type='GELU'),
                 init_cfg=None):
        super(MaxViTBlock, self).__init__(init_cfg=init_cfg)

        self.mbconv = MBConv(in_channels=dim,
                             out_channels=dim_out,
                             stride=stride,
                             expand_ratio=expand_ratio_conv,
                             conv_cfg=conv_cfg,
                             act_cfg=act_cfg_conv,
                             norm_cfg=norm_cfg_conv,
                             drop_path=drop_path)

        self.attn_block = PartitionAttentionCl(dim=dim_out,
                                               partition_type='block',
                                               window_size=window_size,
                                               grid_size=grid_size,
                                               dim_head=dim_head,
                                               attn_bias=attn_bias,
                                               head_first=head_first,
                                               attn_drop=attn_drop,
                                               proj_drop=proj_drop,
                                               ffn_expand_ratio=ffn_expand_ratio,
                                               ffn_drop=ffn_drop,
                                               drop_path=drop_path,
                                               norm_cfg=norm_cfg_transformer,
                                               act_cfg=act_cfg_transformer)

        self.attn_grid = PartitionAttentionCl(dim=dim_out,
                                              partition_type='grid',
                                              window_size=window_size,
                                              grid_size=grid_size,
                                              dim_head=dim_head,
                                              attn_bias=attn_bias,
                                              head_first=head_first,
                                              attn_drop=attn_drop,
                                              proj_drop=proj_drop,
                                              ffn_expand_ratio=ffn_expand_ratio,
                                              ffn_drop=ffn_drop,
                                              drop_path=drop_path)

    def forward(self, x):
        # NCHW format
        x = self.mbconv(x)
        x = x.permute(0, 2, 3, 1)  # to NHWC (channels-last)
        x = self.attn_block(x)
        x = self.attn_grid(x)
        x = x.permute(0, 3, 1, 2)  # back to NCHW
        return x


class Stem(BaseModule):
    """ A Stem layer in MaxViT module.

    A PyTorch implementation of Stem layer introduced by:
    `MaxViT: Multi-Axis Vision Transformer
    <https://arxiv.org/pdf/2204.01697.pdf>`_

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of output channels.
        kernel_size (int): The kernel size of the convolution layer.
            Defaults to 3.
        stride (int): The stride of the first convolution layer.
            Defaults to 2.
        conv_cfg (Tuple[dict]): Config dict for convolution layer.
            Defaults to (dict(type='Conv2d'),dict(type='Conv2dAdaptivePadding')),
            which means using two different conv modules.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', eps=1e-3).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='GELU').
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 conv_cfg=(dict(type='Conv2d'), dict(type='Conv2dAdaptivePadding')),
                 norm_cfg=dict(type='BN', eps=1e-3),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(Stem,self).__init__(init_cfg=init_cfg)

        self.conv1 = ConvModule(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                bias=True,
                                conv_cfg=conv_cfg[1],
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.conv2 = ConvModule(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=1,
                                conv_cfg=conv_cfg[0],
                                norm_cfg=None,
                                act_cfg=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


@MODELS.register_module()
class MaxViT(BaseBackbone):
    """ MaxViT Backbone.

    A PyTorch implementation of MaxViT Backbone introduced by:
    `MaxViT: Multi-Axis Vision Transformer
    <https://arxiv.org/pdf/2204.01697.pdf>`_

    Args:
        embed_dim (Tuple[int]): The embedding dimensions in four stages.
            Defaults to (64, 128, 256, 512). Means maxvit_tiny.
        depths (Tuple[int]): The number of layers in for stages.
            Defaults to (2, 2, 5, 2). Means maxvit_tiny.
        img_size (Union[int, Tuple[int, int]]): The size of input image.
            Defaults to 224.
        in_channels (int): The channels of input image. Defaults to 3.
        stem_width (int): The output channels in the stem layer. Defaults to 64,
        head_hidden_size (int): The hidden channels in final MLP layer, is also the
            output channels in MaxViT backbone. Defaults to 512. Means maxvit_tiny.
        kernel_size (int): The kernel size of the convolution layer in stem layer.
            Defaults to 3.
        stride (int) : Stride of the convolution layer in MBConv layer.
            Defaults to 1.
        expand_ratio_conv (float) : Expend ratio in mid_channels in MBConv layer.
            The mid_channel will be ``make_divisible(channels * ratio, divisor)``.
            Defaults to 4.0
        conv_cfg (Tuple[dict]): Config dict for convolution layer in MBConv layer.
            Defaults to (dict(type='Conv2d'),dict(type='Conv2dAdaptivePadding')),
            which means using two different conv modules.
        norm_cfg_conv (dict): Config dict for normalization layer in MBConv layer.
            Defaults to dict(type='BN', eps=1e-3).
        act_cfg_conv (dict): Config dict for activation layer in MBConv layer.
            Defaults to dict(type='GELU').
        partition_ratio (int): In order to compute window_size and grid_size.
            Defaults to 32.
        dim_head (int): In order to compute the num_heads and scales in
            Channel-last Attention layers. Defaults to 32.
        attn_bias (bool): The bias in the Linear layer in Channel-last Attention layer.
            Defaults to True.
        head_first (bool): Whether to put ``num_heads`` first in
            Channel-last Attention layer. Defaults to False,
        attn_drop (float): The drop out rate for attention output weights in
            Channel-last Attention layer. Defaults to 0.
        proj_drop (float): The drop out rate for Linear output weights in
            Channel-last Attention layer. Defaults to 0.
        ffn_expand_ratio (float) : Expend ratio in FFN layer in Channel-last Attention layer,
            the mid_channel will be ``dim * ffn_expand_ratio``. Defaults to 4.0
        ffn_drop (float): The drop out rate for FFN output weights in
            Channel-last Attention layer. Defaults to 0.
        norm_cfg_transformer (dict): Config dict for normalization layer in
            Channel-last Attention layer. Defaults to dict(type='LN', eps=1e-5).
        act_cfg_transformer (dict): Config dict for activation layer in
            Channel-last Attention layer. Defaults to dict(type='GELU').
        drop_path_rate (float): The ratio of the stochastic depth.
            Defaults to 0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (-1, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 embed_dim: Tuple[int] = (64, 128, 256, 512),
                 depths: Tuple[int] = (2, 2, 5, 2),
                 img_size: Union[int, Tuple[int, int]] = 224,
                 in_channels: int = 3,
                 stem_width: int = 64,
                 head_hidden_size: int = 512,
                 kernel_size: int = 3,
                 stride: int = 2,
                 expand_ratio_conv: float = 4.,
                 conv_cfg=(dict(type='Conv2d'), dict(type='Conv2dAdaptivePadding')),
                 norm_cfg_conv=dict(type='BN', eps=1e-3),
                 act_cfg_conv=dict(type='GELU'),
                 partition_ratio: int = 32,
                 dim_head: int = 32,
                 attn_bias: bool = True,
                 head_first: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 ffn_expand_ratio: float = 4.,
                 ffn_drop: float = 0.,
                 norm_cfg_transformer=dict(type='LN', eps=1e-5),
                 act_cfg_transformer=dict(type='GELU'),
                 drop_path_rate: float = 0.,
                 out_indices: Sequence[int] = (-1,),
                 frozen_stages: int = 0,
                 init_cfg=dict(type='TruncNormal', layer='Linear')):
        super(MaxViT, self).__init__(init_cfg=init_cfg)

        img_size = to_2tuple(img_size)
        self.embed_dim = embed_dim[-1]
        self.head_hidden_size = head_hidden_size

        window_size = (img_size[0] // partition_ratio, img_size[1] // partition_ratio)
        grid_size = (img_size[0] // partition_ratio, img_size[1] // partition_ratio)

        self.stem = Stem(in_channels=in_channels,
                         out_channels=stem_width,
                         kernel_size=kernel_size,
                         stride=stride,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg_conv,
                         act_cfg=act_cfg_conv)

        self.layers = nn.ModuleList()
        self.num_layers = sum(depths)
        dpr = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, self.num_layers).split(depths)
        ]  # stochastic depth decay rule
        in_channels = stem_width
        for i in range(len(depths)):
            for j in range(depths[i]):
                stride = 2 if j == 0 else 1
                out_channels = embed_dim[i]
                self.layers.append(
                    MaxViTBlock(
                        dim=in_channels,
                        dim_out=out_channels,
                        stride=stride,
                        expand_ratio_conv=expand_ratio_conv,
                        conv_cfg=conv_cfg,
                        act_cfg_conv=act_cfg_conv,
                        norm_cfg_conv=norm_cfg_conv,
                        window_size=window_size,
                        grid_size=grid_size,
                        dim_head=dim_head,
                        attn_bias=attn_bias,
                        head_first=head_first,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        ffn_expand_ratio=ffn_expand_ratio,
                        ffn_drop=ffn_drop,
                        norm_cfg_transformer=norm_cfg_transformer,
                        act_cfg_transformer=act_cfg_transformer,
                        drop_path=dpr[i][j]
                    ))
                in_channels = out_channels

        # self.final_norm = LayerNorm2d(self.embed_dim)
        if self.head_hidden_size:
            self.final_mlp = Sequential(nn.AdaptiveAvgPool2d(1),
                                        LayerNorm2d(self.embed_dim),
                                        nn.Flatten(1),
                                        nn.Linear(self.embed_dim,self.head_hidden_size),
                                        build_activation_layer(dict(type='Tanh')))

        # Transform out_indices
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + 1 + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

        if frozen_stages > self.num_layers + 1:
            raise ValueError('frozen_stages must be less than '
                             f'{self.num_layers} but get {frozen_stages}')
        self.frozen_stages = frozen_stages


    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


    def init_weights(self):
        super(MaxViT,self).init_weights()

        if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
            return


    def forward(self, x: torch.Tensor):
        outs = []

        x = self.stem(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if self.head_hidden_size:
            x = self.final_mlp(x)
            if self.num_layers in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

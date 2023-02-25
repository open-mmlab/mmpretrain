# Copyright (c) OpenMMLab. All rights reserved.
from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch
import torch.nn as nn

from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.bricks import ConvModule, DropPath
from mmengine.model import BaseModule, Sequential
from mmengine.model.weight_init import trunc_normal_


from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.utils import SELayer, make_divisible
from mmcls.registry import MODELS
from ..utils import build_norm_layer, to_2tuple


class MBConv(BaseModule):
    """ MBConv block as described in: https://arxiv.org/pdf/2204.01697.pdf.
        Without downsampling:
        x ← x + Proj(SE(DWConv(Conv(Norm(x)))))
        With downsampling:
        x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).
        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.
        Note: This implementation differs slightly from the original MobileNet implementation!
              This implementation differs slightly from the original EfficientnetV2 implementation!
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true downscale by a factor of two is performed. Default: False
        act_cfg (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_cfg (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride=1,
                 expand_ratio=4.0,
                 drop_path: float = 0.,
                 conv_cfg=(dict(type='Conv2d'),dict(type='Conv2dAdaptivePadding')),
                 act_cfg=dict(type='GELU', approximate='tanh'),
                 norm_cfg=dict(type='BN', eps=1e-3),
                 init_cfg=None):
        """ Constructor method """
        # Call super constructor
        super(MBConv, self).__init__(init_cfg=init_cfg)
        # Save parameter

        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        # Check parameters for downscaling
        if stride == 1:
            assert in_channels == out_channels, \
                "If stride is 1, input and output channels must be equal."
        # Ignore inplace parameter if GELU is used
        # Make main path

        if stride == 2:
            self.shortcut = Sequential(
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                ConvModule(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           conv_cfg=conv_cfg[0],
                           norm_cfg=None,
                           act_cfg=None)
            )
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
                                   ratio=4,
                                   conv_cfg=None,
                                   act_cfg=(dict(type='SiLU'),dict(type='Sigmoid'))))

        self.layers.append(ConvModule(in_channels=mid_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      conv_cfg=conv_cfg[0],
                                      norm_cfg=None,
                                      act_cfg=None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
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
        width_lookup: torch.Tensor,) -> torch.Tensor:
    """Reindex 2d relative position bias with 2 independent einsum lookups.

    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py

    Args:
        relative_position_tensor: tensor of shape
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


class RelPosBiasTf(BaseModule):
    """ Relative Position Bias Impl (Compatible with Tensorflow MaxViT models)
    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py
    """
    def __init__(self, window_size, num_heads, prefix_tokens=0):
        super().__init__()
        assert prefix_tokens <= 1
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
        nn.init.normal_(self.relative_position_bias_table, std=.02)

    def get_bias(self) -> torch.Tensor:
        # FIXME change to not use one-hot/einsum?
        return reindex_2d_einsum_lookup(
            self.relative_position_bias_table,
            self.window_size[0],
            self.window_size[1],
            self.height_lookup,
            self.width_lookup
        )

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor] = None):
        return attn + self.get_bias()


class AttentionCl(nn.Module):
    """ Channels-last multi-head attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            dim_head: int = 32,
            bias: bool = True,
            expand_first: bool = True,
            head_first: bool = True,
            rel_pos_cls: Callable = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first and dim_out > dim else dim
        assert dim_attn % dim_head == 0, 'attn dim should be divisible by head_dim'
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim_attn * 3, bias=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
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

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rel_pos is not None:
            attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
        elif shared_rel_pos is not None:
            attn = attn + shared_rel_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size: List[int]):
    B, H, W, C = x.shape
    assert H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})'
    assert W % window_size[1] == 0, ''
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


# @register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition(x, grid_size: List[int]):
    B, H, W, C = x.shape
    assert H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}'
    assert W % grid_size[1] == 0, ''
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


# @register_notrace_function  # reason: int argument is a Proxy
def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


class PartitionAttentionCl(BaseModule):
    """ Grid or Block partition + Attn + FFN.
    NxC 'channels last' tensor layout.
    """

    def __init__(
            self,
            dim: int,
            partition_type: str = 'block',
            window_size=(7, 7),
            grid_size=(7, 7),
            dim_head=32,
            attn_bias=True,
            head_first=False,
            attn_drop=0.,
            proj_drop=0.,
            ffn_expand_ratio=4.0,
            ffn_drop=0.,
            drop_path: float = 0.,
            norm_cfg=dict(type='LN', eps=1e-5),
            act_cfg=dict(type='GELU', approximate='tanh'),
            init_cfg=None
    ):
        super(PartitionAttentionCl, self).__init__(init_cfg=init_cfg)

        assert partition_type in {None, 'block', 'grid'}

        self.partition_block = partition_type == 'block'
        self.partition_size = to_2tuple(window_size if self.partition_block else grid_size)
        rel_pos_cls = partial(RelPosBiasTf, window_size=window_size)

        self.norm1 = build_norm_layer(norm_cfg, dim)
        self.attn = AttentionCl(
            dim,
            dim,
            dim_head=dim_head,
            bias=attn_bias,
            head_first=head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

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
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.ffn(self.norm2(x))))
        return x


class MaxVitBlock(BaseModule):
    """ MaxVit conv, window partition + FFN , grid partition + FFN
    """

    def __init__(self,
                 dim,
                 dim_out,
                 stride= 1,
                 expand_ratio_conv=4.0,
                 conv_cfg=(dict(type='Conv2d'), dict(type='Conv2dAdaptivePadding')),
                 act_cfg=dict(type='GELU', approximate='tanh'),
                 norm_cfg_conv=dict(type='BN', eps=1e-3),
                 window_size=(7, 7),
                 grid_size=(7, 7),
                 dim_head=32,
                 attn_bias=True,
                 head_first=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 ffn_expand_ratio=4.,
                 ffn_drop=0.,
                 drop_path: float = 0.,
                 norm_cfg_transformer=dict(type='LN', eps=1e-5),
                 act_cfg_transformer=dict(type='GELU', approximate='tanh'),
                 init_cfg=None) -> nn.Module:

        super(MaxVitBlock, self).__init__(init_cfg=init_cfg)

        self.conv = MBConv(in_channels=dim,
                           out_channels=dim_out,
                           stride=stride,
                           expand_ratio=expand_ratio_conv,
                           conv_cfg=conv_cfg,
                           act_cfg=act_cfg,
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
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # to NHWC (channels-last)
        x = self.attn_block(x)
        x = self.attn_grid(x)
        x = x.permute(0, 3, 1, 2)  # back to NCHW
        return x


class Stem(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride=2,
                 conv_cfg=(dict(type='Conv2d'), dict(type='Conv2dAdaptivePadding')),
                 norm_cfg=dict(type='BN', eps=1e-3),
                 act_cfg=dict(type='GELU', approximate='tanh'),
                 init_cfg=None):
        super(Stem,self).__init__(init_cfg=init_cfg)

        self.conv1 = ConvModule(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                conv_cfg=conv_cfg[1],
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.conv2 = ConvModule(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=1,
                                conv_cfg=conv_cfg[0],
                                norm_cfg=None,
                                act_cfg=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MaxxVitStage(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 2,
            depth: int = 4,
            feat_size: Tuple[int, int] = (14, 14),
            block_types: Union[str, Tuple[str]] = 'C',
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: Union[float, List[float]] = 0.,
    ):
        super().__init__()
        self.grad_checkpointing = False

        block_types = extend_tuple(block_types, depth)
        blocks = []
        for i, t in enumerate(block_types):
            block_stride = stride if i == 0 else 1
            assert t in ('C', 'T', 'M', 'PM')
            if t == 'C':
                conv_cls = ConvNeXtBlock if conv_cfg.block_type == 'convnext' else MbConvBlock
                blocks += [conv_cls(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    cfg=conv_cfg,
                    drop_path=drop_path[i],
                )]
            elif t == 'T':
                rel_pos_cls = get_rel_pos_cls(transformer_cfg, feat_size)
                blocks += [TransformerBlock2d(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    rel_pos_cls=rel_pos_cls,
                    cfg=transformer_cfg,
                    drop_path=drop_path[i],
                )]
            elif t == 'M':
                blocks += [MaxxVitBlock(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    conv_cfg=conv_cfg,
                    transformer_cfg=transformer_cfg,
                    drop_path=drop_path[i],
                )]
            elif t == 'PM':
                blocks += [ParallelMaxxVitBlock(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    conv_cfg=conv_cfg,
                    transformer_cfg=transformer_cfg,
                    drop_path=drop_path[i],
                )]
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x



class MaxxVit(BaseModule):
    """ CoaTNet + MaxVit base model.

    Highly configurable for different block compositions, tensor layouts, pooling types.
    """

    def __init__(
            self,
            cfg: MaxxVitCfg,
            embed_dim=(64, 128, 256, 512),
            depths=(2, 2, 5, 2),
            img_size: Union[int, Tuple[int, int]] = 224,
            in_channels: int = 3,
            stem_width=64,
            kernel_size=3,
            stride=2,
            conv_cfg=(dict(type='Conv2d'), dict(type='Conv2dAdaptivePadding')),
            norm_cfg=dict(type='BN', eps=1e-3),
            act_cfg=dict(type='GELU', approximate='tanh'),
            # num_classes: int = 1000,
            # global_pool: str = 'avg',
            # drop_rate: float = 0.,
            drop_path_rate: float = 0.
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        transformer_cfg = cfg_window_size(cfg.transformer_cfg, img_size)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = cfg.embed_dim[-1]
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        self.stem = Stem(in_channels=in_channels,
                         out_channels=stem_width,
                         kernel_size=kernel_size,
                         stride=stride,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg)

        self.layers = nn.ModuleList()
        in_channels = stem_width
        for i in len(depths):
            for j in range(depths[i]):
                stride = 2 if j == 0 else 1
                out_channels = embed_dim[i]
                self.layers.append(
                    MaxVitBlock(
                        dim=in_channels,
                        dim_out=out_channels,
                        stride=stride,
                        expand_ratio_conv=4.,
                        conv_cfg=(dict(type='Conv2d'), dict(type='Conv2dAdaptivePadding')),
                        act_cfg=dict(type='GELU', approximate='tanh'),
                        norm_cfg_conv=dict(type='BN', eps=1e-3),
                        window_size=(7, 7),
                        grid_size=(7, 7),
                        dim_head=32,
                        attn_bias=True,
                        head_first=False,
                        attn_drop=0.,
                        proj_drop=0.,
                        ffn_expand_ratio=4.,
                        ffn_drop=0.,
                        drop_path: float = 0.,
                        norm_cfg_transformer=dict(type='LN', eps=1e-5),
                        act_cfg_transformer=dict(type='GELU', approximate='tanh')))

        self._make_layers()


        stride = self.stem.stride
        feat_size = tuple([i // s for i, s in zip(img_size, to_2tuple(stride))])

        num_stages = len(cfg.embed_dim)
        assert len(cfg.depths) == num_stages
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        in_chs = self.stem.out_chs
        stages = []
        for i in range(num_stages):
            stage_stride = 2
            out_chs = cfg.embed_dim[i]
            feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
            stages += [MaxxVitStage(
                in_chs,
                out_chs,
                depth=cfg.depths[i],
                block_types=cfg.block_type[i],
                conv_cfg=cfg.conv_cfg,
                transformer_cfg=transformer_cfg,
                feat_size=feat_size,
                drop_path=dpr[i],
            )]
            stride *= stage_stride
            in_chs = out_chs
        self.stages = nn.Sequential(*stages)

        final_norm_layer = partial(get_norm_layer(cfg.transformer_cfg.norm_layer), eps=cfg.transformer_cfg.norm_eps)
        if cfg.head_hidden_size:
            self.norm = nn.Identity()
            self.head = NormMlpHead(
                self.num_features,
                num_classes,
                hidden_size=cfg.head_hidden_size,
                pool_type=global_pool,
                drop_rate=drop_rate,
                norm_layer=final_norm_layer,
            )
        else:
            # standard classifier head w/ norm, pooling, fc classifier
            self.norm = final_norm_layer(self.num_features)
            self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        # Weight init (default PyTorch init works well for AdamW if scheme not set)
        assert cfg.weight_init in ('', 'normal', 'trunc_normal', 'xavier_normal', 'vit_eff')
        if cfg.weight_init:
            named_apply(partial(self._init_weights, scheme=cfg.weight_init), self)

    def _make_layers(self):
        pass

    def _init_weights(self, module, name, scheme=''):
        if hasattr(module, 'init_weights'):
            try:
                module.init_weights(scheme=scheme)
            except TypeError:
                module.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.head.global_pool.pool_type
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

maxvit_model_cfgs = dict(
    # Trying to be like the MaxViT paper configs
    maxvit_tiny_tf=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        stem_bias=True,
        head_hidden_size=512,
        **_tf_cfg(),
    ),
    maxvit_small_tf=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        stem_bias=True,
        head_hidden_size=768,
        **_tf_cfg(),
    ),
    maxvit_base_tf=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=64,
        stem_bias=True,
        head_hidden_size=768,
        **_tf_cfg(),
    ),
    maxvit_large_tf=MaxxVitCfg(
        embed_dim=(128, 256, 512, 1024),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=128,
        stem_bias=True,
        head_hidden_size=1024,
        **_tf_cfg(),
    ),
    maxvit_xlarge_tf=MaxxVitCfg(
        embed_dim=(192, 384, 768, 1536),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=192,
        stem_bias=True,
        head_hidden_size=1536,
        **_tf_cfg(),
    ),
)
#
#
# def window_partition(input: torch.Tensor,
#                      window_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
#     """ Window partition function.
#     Args:
#         input (torch.Tensor): Input tensor of the shape [B, C, H, W].
#         window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)
#     Returns:
#         windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
#     """
#     # Get size of input
#     B, C, H, W = input.shape
#     # Unfold input
#     windows = input.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
#     # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
#     windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
#     return windows
#
#
# def window_reverse(
#         windows: torch.Tensor,
#         original_size: Tuple[int, int],
#         window_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
#     """ Reverses the window partition.
#     Args:
#         windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
#         original_size (Tuple[int, int]): Original shape.
#         window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
#     Returns:
#         output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
#     """
#     # Get height and width
#     H, W = original_size
#     # Compute original batch size
#     B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
#     # Fold grid tensor
#     output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
#     output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
#     return output
#
#
# def grid_partition(
#         input: torch.Tensor,
#         grid_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
#     """ Grid partition function.
#     Args:
#         input (torch.Tensor): Input tensor of the shape [B, C, H, W].
#         grid_size (Tuple[int, int], optional): Grid size to be applied. Default (7, 7)
#     Returns:
#         grid (torch.Tensor): Unfolded input tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
#     """
#     # Get size of input
#     B, C, H, W = input.shape
#     # Unfold input
#     grid = input.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
#     # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], window_size[1], C]
#     grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
#     return grid
#
#
# def grid_reverse(
#         grid: torch.Tensor,
#         original_size: Tuple[int, int],
#         grid_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
#     """ Reverses the grid partition.
#     Args:
#         Grid (torch.Tensor): Grid tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
#         original_size (Tuple[int, int]): Original shape.
#         grid_size (Tuple[int, int], optional): Grid size which have been applied. Default (7, 7)
#     Returns:
#         output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
#     """
#     # Get height, width, and channels
#     (H, W), C = original_size, grid.shape[-1]
#     # Compute original batch size
#     B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
#     # Fold grid tensor
#     output = grid.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
#     output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
#     return output
#
#
# def get_relative_position_index(win_h: int, win_w: int) -> torch.Tensor:
#     """ Function to generate pair-wise relative position index for each token inside the window.
#         Taken from Timms Swin V1 implementation.
#     Args:
#         win_h (int): Window/Grid height.
#         win_w (int): Window/Grid width.
#     Returns:
#         relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
#     """
#     coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
#     coords_flatten = torch.flatten(coords, 1)
#     relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#     relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#     relative_coords[:, :, 0] += win_h - 1
#     relative_coords[:, :, 1] += win_w - 1
#     relative_coords[:, :, 0] *= 2 * win_w - 1
#     return relative_coords.sum(-1)
#
#
# class RelativeSelfAttention(BaseModule):
#     """ Relative Self-Attention similar to Swin V1. Implementation inspired by Timms Swin V1 implementation.
#     Args:
#         in_channels (int): Number of input channels.
#         num_heads (int, optional): Number of attention heads. Default 32
#         grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         drop (float, optional): Dropout ratio of output. Default: 0.0
#     """
#
#     def __init__(
#             self,
#             in_channels: int,
#             num_heads: int = 32,
#             grid_window_size: Tuple[int, int] = (7, 7),
#             attn_drop: float = 0.,
#             drop: float = 0.,
#             init_cfg=None
#     ) -> None:
#         """ Constructor method """
#         # Call super constructor
#         super(RelativeSelfAttention, self).__init__(init_cfg=init_cfg)
#         # Save parameters
#         self.in_channels: int = in_channels
#         self.num_heads: int = num_heads
#         self.grid_window_size: Tuple[int, int] = grid_window_size
#         self.scale: float = num_heads ** -0.5
#         self.attn_area: int = grid_window_size[0] * grid_window_size[1]
#         # Init layers
#         self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
#         self.attn_drop = nn.Dropout(p=attn_drop)
#         self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
#         self.proj_drop = nn.Dropout(p=drop)
#         self.softmax = nn.Softmax(dim=-1)
#         # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads))
#
#         # Get pair-wise relative position index for each token inside the window
#         self.register_buffer("relative_position_index", get_relative_position_index(grid_window_size[0],
#                                                                                     grid_window_size[1]))
#         # Init relative positional bias
#         trunc_normal_(self.relative_position_bias_table, std=.02)
#
#     def _get_relative_positional_bias(self) -> torch.Tensor:
#         """ Returns the relative positional bias.
#         Returns:
#             relative_position_bias (torch.Tensor): Relative positional bias.
#         """
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#         return relative_position_bias.unsqueeze(0)
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         """ Forward pass.
#         Args:
#             input (torch.Tensor): Input tensor of the shape [B_, N, C].
#         Returns:
#             output (torch.Tensor): Output tensor of the shape [B_, N, C].
#         """
#         # Get shape of input
#         B_, N, C = input.shape
#         # Perform query key value mapping
#         qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         # Scale query
#         q = q * self.scale
#         # Compute attention maps
#         attn = self.softmax(q @ k.transpose(-2, -1) + self._get_relative_positional_bias())
#         # Map value with attention maps
#         output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
#         # Perform final projection and dropout
#         output = self.proj(output)
#         output = self.proj_drop(output)
#         return output
#
#
# class MaxViTTransformerBlock(BaseModule):
#     """ MaxViT Transformer block.
#         With block partition:
#         x ← x + Unblock(RelAttention(Block(LN(x))))
#         x ← x + MLP(LN(x))
#         With grid partition:
#         x ← x + Ungrid(RelAttention(Grid(LN(x))))
#         x ← x + MLP(LN(x))
#         Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
#         Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.
#     Args:
#         in_channels (int): Number of input channels.
#         partition_function (Callable): Partition function to be utilized (grid or window partition).
#         reverse_function (Callable): Reverse function to be utilized  (grid or window reverse).
#         num_heads (int, optional): Number of attention heads. Default 32
#         grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         drop (float, optional): Dropout ratio of output. Default: 0.0
#         drop_path (float, optional): Dropout ratio of path. Default: 0.0
#         mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
#         act_cfg (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
#         norm_cfg (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  partition_function: Callable,
#                  reverse_function: Callable,
#                  num_heads: int = 32,
#                  grid_window_size: Tuple[int, int] = (7, 7),
#                  attn_drop: float = 0.,
#                  drop: float = 0.,
#                  drop_path: float = 0.,
#                  mlp_ratio: float = 4.,
#                  act_cfg=dict(type='GELU'),
#                  norm_cfg=dict(type='LN'),
#                  init_cfg=None) -> None:
#         """ Constructor method """
#         super(MaxViTTransformerBlock, self).__init__(init_cfg=init_cfg)
#         # Save parameters
#         self.partition_function: Callable = partition_function
#         self.reverse_function: Callable = reverse_function
#         self.grid_window_size: Tuple[int, int] = grid_window_size
#         # Init layers
#         self.norm1 = build_norm_layer(norm_cfg, in_channels)
#         self.attention = RelativeSelfAttention(
#             in_channels=in_channels,
#             num_heads=num_heads,
#             grid_window_size=grid_window_size,
#             attn_drop=attn_drop,
#             drop=drop
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = build_norm_layer(norm_cfg, in_channels)
#
#         self.mlp = FFN(embed_dims=in_channels,
#                        feedforward_channels=int(mlp_ratio * in_channels),
#                        act_cfg=act_cfg,
#                        ffn_drop=drop)
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         """ Forward pass.
#         Args:
#             input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].
#         Returns:
#             output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)].
#         """
#         # Save original shape
#         B, C, H, W = input.shape
#         # Perform partition
#         input_partitioned = self.partition_function(input, self.grid_window_size)
#         input_partitioned = input_partitioned.view(-1, self.grid_window_size[0] * self.grid_window_size[1], C)
#         # Perform normalization, attention, and dropout
#         output = input_partitioned + self.drop_path(self.attention(self.norm1(input_partitioned)))
#         # Perform normalization, MLP, and dropout
#         output = output + self.drop_path(self.mlp(self.norm2(output)))
#         # Reverse partition
#         output = self.reverse_function(output, (H, W), self.grid_window_size)
#         return output
#
#
# class MaxViTBlock(BaseModule):
#     """ MaxViT block composed of MBConv block, Block Attention, and Grid Attention.
#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         downscale (bool, optional): If true spatial downscaling is performed. Default: False
#         num_heads (int, optional): Number of attention heads. Default 32
#         grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         drop (float, optional): Dropout ratio of output. Default: 0.0
#         drop_path (float, optional): Dropout ratio of path. Default: 0.0
#         mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
#         act_cfg (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
#         norm_cfg (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
#         norm_cfg_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
#     """
#
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             stride=1,
#             num_heads: int = 32,
#             grid_window_size: Tuple[int, int] = (7, 7),
#             attn_drop: float = 0.,
#             drop: float = 0.,
#             drop_path: float = 0.,
#             mlp_ratio: float = 4.,
#             act_cfg=dict(type='GELU'),
#             norm_cfg=dict(type='BN'),
#             norm_cfg_transformer=dict(type='LN'),
#             init_cfg=None) -> None:
#         """ Constructor method """
#         # Call super constructor
#         super(MaxViTBlock, self).__init__(init_cfg=init_cfg)
#         # Init MBConv block
#         self.mb_conv = MBConv(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             stride=stride,
#             drop_path=drop_path,
#             norm_cfg=norm_cfg
#         )
#         # Init Block and Grid Transformer
#         self.block_transformer = MaxViTTransformerBlock(
#             in_channels=out_channels,
#             partition_function=window_partition,
#             reverse_function=window_reverse,
#             num_heads=num_heads,
#             grid_window_size=grid_window_size,
#             attn_drop=attn_drop,
#             drop=drop,
#             drop_path=drop_path,
#             mlp_ratio=mlp_ratio,
#             act_cfg=act_cfg,
#             norm_cfg=norm_cfg_transformer
#         )
#         self.grid_transformer = MaxViTTransformerBlock(
#             in_channels=out_channels,
#             partition_function=grid_partition,
#             reverse_function=grid_reverse,
#             num_heads=num_heads,
#             grid_window_size=grid_window_size,
#             attn_drop=attn_drop,
#             drop=drop,
#             drop_path=drop_path,
#             mlp_ratio=mlp_ratio,
#             act_cfg=act_cfg,
#             norm_cfg=norm_cfg_transformer
#         )
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         """ Forward pass.
#         Args:
#             input (torch.Tensor): Input tensor of the shape [B, C_in, H, W]
#         Returns:
#             output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2] (downscaling is optional)
#         """
#         output = self.grid_transformer(self.block_transformer(self.mb_conv(input)))
#         return output
#
#
# class MaxViTStage(BaseModule):
#     """ Stage of the MaxViT.
#     Args:
#         depth (int): Depth of the stage.
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         num_heads (int, optional): Number of attention heads. Default 32
#         grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         drop (float, optional): Dropout ratio of output. Default: 0.0
#         drop_path (float, optional): Dropout ratio of path. Default: 0.0
#         mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
#         act_cfg (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
#         norm_cfg (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
#         norm_cfg_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
#     """
#
#     def __init__(
#             self,
#             depth: int,
#             in_channels: int,
#             out_channels: int,
#             num_heads: int = 32,
#             grid_window_size: Tuple[int, int] = (7, 7),
#             attn_drop: float = 0.,
#             drop: float = 0.,
#             drop_path: Union[List[float], float] = 0.,
#             mlp_ratio: float = 4.,
#             act_cfg=dict(type='GELU'),
#             norm_cfg=dict(type='BN'),
#             norm_cfg_transformer=dict(type='LN'),
#             init_cfg=None) -> None:
#         """ Constructor method """
#         # Call super constructor
#         super(MaxViTStage, self).__init__(init_cfg=init_cfg)
#         # Init blocks
#         self.blocks = nn.Sequential(*[
#             MaxViTBlock(
#                 in_channels=in_channels if index == 0 else out_channels,
#                 out_channels=out_channels,
#                 stride=2 if index == 0 else 1,
#                 num_heads=num_heads,
#                 grid_window_size=grid_window_size,
#                 attn_drop=attn_drop,
#                 drop=drop,
#                 drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
#                 mlp_ratio=mlp_ratio,
#                 act_cfg=act_cfg,
#                 norm_cfg=norm_cfg,
#                 norm_cfg_transformer=norm_cfg_transformer
#             )
#             for index in range(depth)
#         ])
#
#     def forward(self, input=torch.Tensor) -> torch.Tensor:
#         """ Forward pass.
#         Args:
#             input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].
#         Returns:
#             output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2].
#         """
#         output = self.blocks(input)
#         return output
#
# @MODELS.register_module()
# class MaxViT(BaseBackbone):
#     """ Implementation of the MaxViT proposed in:
#         https://arxiv.org/pdf/2204.01697.pdf
#     Args:
#         in_channels (int, optional): Number of input channels to the convolutional stem. Default 3
#         depths (Tuple[int, ...], optional): Depth of each network stage. Default (2, 2, 5, 2)
#         channels (Tuple[int, ...], optional): Number of channels in each network stage. Default (64, 128, 256, 512)
#         num_classes (int, optional): Number of classes to be predicted. Default 1000
#         embed_dim (int, optional): Embedding dimension of the convolutional stem. Default 64
#         num_heads (int, optional): Number of attention heads. Default 32
#         grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         drop (float, optional): Dropout ratio of output. Default: 0.0
#         drop_path (float, optional): Dropout ratio of path. Default: 0.0
#         mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
#         act_cfg (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
#         norm_cfg (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
#         norm_cfg_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
#         global_pool (str, optional): Global polling type to be utilized. Default "avg"
#     """
#
#     def __init__(
#             self,
#             in_channels: int = 3,
#             depths: Tuple[int, ...] = (2, 2, 5, 2),
#             channels: Tuple[int, ...] = (64, 128, 256, 512),
#             num_classes: int = 1000,
#             embed_dim: int = 64,
#             num_heads: int = 32,
#             grid_window_size: Tuple[int, int] = (7, 7),
#             attn_drop: float = 0.,
#             drop=0.,
#             drop_path=0.,
#             mlp_ratio=4.,
#             act_cfg=dict(type='GELU'),
#             norm_cfg=dict(type='BN', eps=1e-5),
#             norm_cfg_transformer=dict(type='LN'),
#             global_pool: str = "avg",
#             init_cfg=None
#     ) -> None:
#         """ Constructor method """
#         # Call super constructor
#         super(MaxViT, self).__init__(init_cfg=init_cfg)
#         # Check parameters
#         assert len(depths) == len(channels), "For each stage a channel dimension must be given."
#         assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
#         # Save parameters
#         self.num_classes: int = num_classes
#         # Init convolutional stem
#         self.stem = nn.Sequential(
#             ConvModule(in_channels=in_channels,
#                        out_channels=embed_dim,
#                        kernel_size=3,
#                        stride=2,
#                        conv_cfg=dict(type='Conv2dAdaptivePadding'),
#                        norm_cfg=norm_cfg,
#                        act_cfg=act_cfg),
#             ConvModule(in_channels=embed_dim,
#                        out_channels=embed_dim,
#                        kernel_size=3,
#                        stride=1,
#                        conv_cfg=dict(type='Conv2d'),
#                        norm_cfg=None,
#                        act_cfg=None)
#         )
#         # Init blocks
#         drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
#         stages = []
#         for index, (depth, channel) in enumerate(zip(depths, channels)):
#             stages.append(
#                 MaxViTStage(
#                     depth=depth,
#                     in_channels=embed_dim if index == 0 else channels[index - 1],
#                     out_channels=channel,
#                     num_heads=num_heads,
#                     grid_window_size=grid_window_size,
#                     attn_drop=attn_drop,
#                     drop=drop,
#                     drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
#                     mlp_ratio=mlp_ratio,
#                     act_cfg=act_cfg,
#                     norm_cfg=norm_cfg,
#                     norm_cfg_transformer=norm_cfg_transformer
#                 )
#             )
#         self.stages = nn.ModuleList(stages)
#         self.global_pool: str = global_pool
#         self.head = nn.Linear(channels[-1], num_classes)
#
#     @torch.jit.ignore
#     def no_weight_decay(self) -> Set[str]:
#         """ Gets the names of parameters to not apply weight decay to.
#         Returns:
#             nwd (Set[str]): Set of parameter names to not apply weight decay to.
#         """
#         nwd = set()
#         for n, _ in self.named_parameters():
#             if "relative_position_bias_table" in n:
#                 nwd.add(n)
#         return nwd
#
#     def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
#         """Method results the classification head
#         Args:
#             num_classes (int): Number of classes to be predicted
#             global_pool (str, optional): If not global pooling is updated
#         """
#         self.num_classes: int = num_classes
#         if global_pool is not None:
#             self.global_pool = global_pool
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
#
#     def forward_features(self, input: torch.Tensor) -> torch.Tensor:
#         """ Forward pass of feature extraction.
#         Args:
#             input (torch.Tensor): Input images of the shape [B, C, H, W].
#         Returns:
#             output (torch.Tensor): Image features of the backbone.
#         """
#         output = input
#         for stage in self.stages:
#             output = stage(output)
#         return output
#
#     def forward_head(self, input: torch.Tensor, pre_logits: bool = False):
#         """ Forward pass of classification head.
#         Args:
#             input (torch.Tensor): Input features
#             pre_logits (bool, optional): If true pre-logits are returned
#         Returns:
#             output (torch.Tensor): Classification output of the shape [B, num_classes].
#         """
#         if self.global_pool == "avg":
#             input = input.mean(dim=(2, 3))
#         elif self.global_pool == "max":
#             input = torch.amax(input, dim=(2, 3))
#         return input if pre_logits else self.head(input)
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         """ Forward pass
#         Args:
#             input (torch.Tensor): Input images of the shape [B, C, H, W].
#         Returns:
#             output (torch.Tensor): Classification output of the shape [B, num_classes].
#         """
#         output = self.forward_features(self.stem(input))
#         output = self.forward_head(output)
#         return output


def max_vit_tiny_224(**kwargs) -> MaxViT:
    """ MaxViT tiny for a resolution of 224 X 224"""
    return MaxViT(
        depths=(2, 2, 5, 2),
        channels=(64, 128, 256, 512),
        embed_dim=64,
        **kwargs
    )


def max_vit_small_224(**kwargs) -> MaxViT:
    """ MaxViT small for a resolution of 224 X 224"""
    return MaxViT(
        depths=(2, 2, 5, 2),
        channels=(96, 128, 256, 512),
        embed_dim=64,
        **kwargs
    )


def max_vit_base_224(**kwargs) -> MaxViT:
    """ MaxViT base for a resolution of 224 X 224"""
    return MaxViT(
        depths=(2, 6, 14, 2),
        channels=(96, 192, 384, 768),
        embed_dim=64,
        **kwargs
    )


def max_vit_large_224(**kwargs) -> MaxViT:
    """ MaxViT large for a resolution of 224 X 224"""
    return MaxViT(
        depths=(2, 6, 14, 2),
        channels=(128, 256, 512, 1024),
        embed_dim=128,
        **kwargs
    )

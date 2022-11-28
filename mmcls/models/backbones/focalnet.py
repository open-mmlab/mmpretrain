# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (Conv2d, ConvModule, build_activation_layer,
                      build_conv_layer, build_norm_layer)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, AdaptivePadding, PatchEmbed
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.utils import to_2tuple
from torch.nn import functional as F

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.registry import MODELS
from ..utils import LayerScale


class FocalModulation(BaseModule):
    """Focal Modulation.

    Args:
        embed_dims (int): The feature dimension
        drop_rate (float): Dropout rate. Defaults to 0.0.
        focal_level (int): Number of focal levels. Defaults to 2.
        focal_window (int): Focal window size at focal level 1. Defaults to 7.
        focal_factor (int): Step to increase the focal window. Defaults to 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        normalize_modulator (bool): Whether to use normalize modulator.
             Defaults to False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to None.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 drop_rate=0.,
                 focal_level=2,
                 focal_window=7,
                 focal_factor=2,
                 act_cfg=dict(type='GELU'),
                 normalize_modulator=False,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.normalize_modulator = normalize_modulator

        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor

        self.f = nn.Linear(
            embed_dims, 2 * embed_dims + (self.focal_level + 1), bias=True)
        self.h = Conv2d(
            embed_dims, embed_dims, 1, stride=1, padding=0, groups=1)

        self.act = build_activation_layer(act_cfg)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)
        self.focal_layers = ModuleList()

        if norm_cfg is not None:
            self.ln = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.ln = None

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                ConvModule(
                    embed_dims,
                    embed_dims,
                    kernel_size,
                    stride=1,
                    groups=embed_dims,
                    padding=kernel_size // 2,
                    bias=False,
                    act_cfg=act_cfg))

    def forward(self, x):
        C = x.shape[-1]
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * gates[:, level:level + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        x_out = self.ln(x_out) if self.ln else x_out
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out


class FocalModulationBlock(BaseModule):
    """Focal Modulation Block.

    Args:
        embed_dims (int): The feature dimension
        ffn_ratio (float): Ratio of ffn hidden dim to embedding dim.
            Defaults to 4.0.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
        focal_level (int): Number of focal levels. Defaults to 2.
        focal_window (int): Focal window size at focal level 1. Defaults to 9.
        use_layer_scale (bool): Whether to use use_layer_scale in
            FocalModulationBlock. Defaults to False.
        use_postln (bool): Whether to use post-LN in FocalModulationBlock.
            If flase, it uses pre-LN. Defaults to False.
        normalize_modulator (bool): Whether to use normalize modulator.
             Defaults to False.
        use_postln_in_modulation (bool): Whether to use LN in FocalModulation.
            Defaults to False.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 focal_level=2,
                 focal_window=9,
                 use_layer_scale=False,
                 use_postln=False,
                 normalize_modulator=False,
                 use_postln_in_modulation=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.ffn_ratio = ffn_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.with_cp = with_cp
        self.use_postln = use_postln

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.modulation = FocalModulation(
            embed_dims,
            focal_window=self.focal_window,
            focal_level=self.focal_level,
            drop_rate=drop_rate,
            normalize_modulator=normalize_modulator,
            norm_cfg=dict(type='LN') if use_postln_in_modulation else None)

        dropout_layer = dict(type='DropPath', drop_prob=drop_path_rate)
        self.drop_path = build_dropout(
            dropout_layer) if drop_path_rate > 0 else nn.Identity()

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False)

        if use_layer_scale:
            self.gamma1 = LayerScale(embed_dims)
            self.gamma2 = LayerScale(embed_dims)
        else:
            self.gamma1, self.gamma2 = nn.Identity(), nn.Identity()

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            B, L, C = x.shape
            H, W = hw_shape
            assert L == H * W, f"The query length {L} doesn't match the input"\
                f' shape ({H}, {W}).'

            shortcut = x
            x = x if self.use_postln else self.ln1(x)
            x = x.view(B, H, W, C)

            x = self.modulation(x).view(B, H * W, C)
            x = x if not self.use_postln else self.ln1(x)

            x = shortcut + self.drop_path(self.gamma1(x))
            if self.use_postln:
                x = x + self.drop_path(self.gamma2(self.ln2(self.ffn(x))))
            else:
                x = x + self.drop_path(self.gamma2(self.ffn(self.ln2(x))))

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class BasicLayer(BaseModule):
    """A basic focal modulation layer for one stage.

    Args:
        embed_dims (int): The feature dimension
        depth (int): The number of blocks in this stage.
        ffn_ratio (float): Ratio of ffn hidden dim to embedding dim.
            Defaults to 4.0.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        drop_paths (Sequence[float] | float): The drop path rate in each block.
            Defaults to 0.0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        downsample (bool): Downsample the output of blocks. Defaults to False.
        downsample_cfg (dict): The extra config of the downsample layer.
            Defaults to empty dict.
        focal_level (int): Number of focal levels. Defaults to 2.
        focal_window (int): Focal window size at focal level 1. Defaults to 9.
        use_overlapped_embed (bool): Whether to use overlapped convolution for
            downsample. Defaults to False.
        use_layer_scale (bool): Whether to use use_layer_scale in
            FocalModulationBlock. Defaults to False.
        use_postln (bool): Whether to use post-LN in FocalModulationBlock.
            If flase, it uses pre-LN. Defaults to False.
        normalize_modulator (bool): Whether to use normalize modulator.
             Defaults to False.
        use_postln_in_modulation (bool): Whether to use LN in FocalModulation.
            Defaults to False.
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 depth,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_paths=0.,
                 norm_cfg=dict(type='LN'),
                 downsample=False,
                 downsample_cfg=dict(),
                 focal_level=2,
                 focal_window=9,
                 use_overlapped_embed=False,
                 use_layer_scale=False,
                 use_postln=False,
                 normalize_modulator=False,
                 use_postln_in_modulation=False,
                 block_cfgs=dict(),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        if not isinstance(block_cfgs, Sequence):
            block_cfgs = [deepcopy(block_cfgs) for _ in range(depth)]

        self.depth = depth
        self.embed_dims = embed_dims
        self.blocks = ModuleList()
        for i in range(depth):
            _block_cfg = {
                'embed_dims': embed_dims,
                'ffn_ratio': ffn_ratio,
                'drop_rate': drop_rate,
                'drop_path_rate': drop_paths[i],
                'focal_level': focal_level,
                'focal_window': focal_window,
                'use_layer_scale': use_layer_scale,
                'norm_cfg': norm_cfg,
                'use_postln': use_postln,
                'with_cp': with_cp,
                'normalize_modulator': normalize_modulator,
                'use_postln_in_modulation': use_postln_in_modulation,
                **block_cfgs[i]
            }
            block = FocalModulationBlock(**_block_cfg)
            self.blocks.append(block)

        if downsample:
            if use_overlapped_embed:
                _downsample_cfg = dict(
                    in_channels=embed_dims,
                    input_size=None,
                    embed_dims=2 * embed_dims,
                    conv_type='Conv2d',
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    norm_cfg=dict(type='LN'),
                    **downsample_cfg)
                self.downsample = OverlappedPatchEmbed(**_downsample_cfg)
            else:
                _downsample_cfg = dict(
                    in_channels=embed_dims,
                    input_size=None,
                    embed_dims=2 * embed_dims,
                    conv_type='Conv2d',
                    kernel_size=2,
                    stride=2,
                    norm_cfg=dict(type='LN'),
                    **downsample_cfg)
                self.downsample = PatchEmbed(**_downsample_cfg)
        else:
            self.downsample = None

    def forward(self, x, in_shape, do_downsample=True):
        for blk in self.blocks:
            x = blk(x, in_shape)
        if self.downsample is not None and do_downsample:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, *in_shape)
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape
        return x, out_shape

    @property
    def out_channels(self):
        if self.downsample:
            return self.downsample.embed_dims
        else:
            return self.embed_dims


class OverlappedPatchEmbed(PatchEmbed):
    """Image to Patch Embedding with overlapped convolution.

    The differences between OverlappedPatchEmbed & PatchEmbed:
        1. Use adaptive_padding & padding in projection for overlapped
           convolution.

    Args:
        in_channels (int): The num of input channels. Defaults to 3.
        embed_dims (int): The dimensions of embedding. Defaults to 768.
        conv_type (str): The type of convolution
            to generate patch embedding. Defaults to "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Defaults to 7.
        stride (int): The slide stride of embedding conv.
            Defaults to 4.
        padding (int | tuple): The padding length of embedding conv.
            Defaults to 3.
        dilation (int): The dilation rate of embedding conv. Defaults to 1.
        bias (bool): Bias of embed conv. Defaults to True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Defaults to None.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=7,
                 stride=4,
                 padding=3,
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.adaptive_padding = AdaptivePadding(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding='corner')
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None


@MODELS.register_module()
class FocalNet(BaseBackbone):
    """FocalNet.

    A PyTorch implement of :
    `Focal Modulation Networks <https://arxiv.org/abs/2203.11926>`_

    Inspiration from
    https://github.com/microsoft/FocalNet

    Args:
        arch (str | dict): FocalNet architecture. If use string, choose
            from 'tiny-srf', 'tiny-lrf', 'small-srf', 'small-lrf', 'base-srf',
            'base-lrf', 'large-fl3', 'large-fl4', 'xlarge-fl3', 'xlarge-fl4',
            'huge-fl3' and 'huge-fl4'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **ffn_ratio** (float): The ratio of ffn hidden dim to embedding
                dim.
            - **depths** (List[int]): The number of blocks in each stage.
            - **focal_levels** (List[int]): The number of focal levels of each
                stage.
            - **focal_windows** (List[int]): The number of focal window sizes
                at first focal level of each stage.
            - **use_overlapped_embed** (bool): Whether to use overlapped
                convolution for patch embed and downsample.
            - **use_postln** (bool): WWhether to use post-LN in
                FocalModulationBlock. If flase, it uses pre-LN.
            - **use_layer_scale** (bool): Whether to use use_layer_scale.
            - **normalize_modulator** (bool): Whether to use normalize
                modulator.
            - **use_postln_in_modulation** (bool): Whether to use LN in
                FocalModulation.

            Defaults to 't-srf'.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        in_channels (int): The num of input channels. Defaults to 3.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        drop_paths (Sequence[float] | float): The drop path rate in each block.
            Defaults to 0.0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        out_after_downsample (bool): Whether to output the feature map of a
            stage after the following downsample layer. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        stage_cfgs (Sequence[dict] | dict): Extra config dict for each
            stage. Defaults to an empty dict.
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to an empty dict.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['t-srf', 'tiny-srf'], {
                'embed_dims': 96,
                'ffn_ratio': 4.,
                'depths': [2, 2, 6, 2],
                'focal_levels': [2, 2, 2, 2],
                'focal_windows': [3, 3, 3, 3],
                'use_overlapped_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['t-lrf', 'tiny-lrf'], {
                'embed_dims': 96,
                'ffn_ratio': 4.,
                'depths': [2, 2, 6, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [3, 3, 3, 3],
                'use_overlapped_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['s-srf', 'small-srf'], {
                'embed_dims': 96,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [2, 2, 2, 2],
                'focal_windows': [3, 3, 3, 3],
                'use_overlapped_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['s-lrf', 'small-lrf'], {
                'embed_dims': 96,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [3, 3, 3, 3],
                'use_overlapped_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['b-srf', 'base-srf'], {
                'embed_dims': 128,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [2, 2, 2, 2],
                'focal_windows': [3, 3, 3, 3],
                'use_overlapped_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['b-lrf', 'base-lrf'], {
                'embed_dims': 128,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [3, 3, 3, 3],
                'use_overlapped_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['l-fl3', 'large-fl3'], {
                'embed_dims': 192,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [5, 5, 5, 5],
                'use_overlapped_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['l-fl4', 'large-fl4'], {
                'embed_dims': 192,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [4, 4, 4, 4],
                'focal_windows': [3, 3, 3, 3],
                'use_overlapped_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': True,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['xl-fl3', 'xlarge-fl3'], {
                'embed_dims': 256,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [5, 5, 5, 5],
                'use_overlapped_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['xl-fl4', 'xlarge-fl4'], {
                'embed_dims': 256,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [4, 4, 4, 4],
                'focal_windows': [3, 3, 3, 3],
                'use_overlapped_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['h-fl3', 'huge-fl3'], {
                'embed_dims': 352,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [3, 3, 3, 3],
                'use_overlapped_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': False,
                'use_postln_in_modulation': True,
            }),
        **dict.fromkeys(
            ['h-fl4', 'huge-fl4'], {
                'embed_dims': 352,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [4, 4, 4, 4],
                'focal_windows': [3, 3, 3, 3],
                'use_overlapped_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': False,
                'use_postln_in_modulation': True,
            }),
    }

    def __init__(self,
                 arch='t-srf',
                 patch_size=4,
                 in_channels=3,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_cfg=dict(type='LN'),
                 out_indices=(3, ),
                 out_after_downsample=False,
                 frozen_stages=-1,
                 with_cp=False,
                 stage_cfgs=dict(),
                 patch_cfg=dict(),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'ffn_ratio', 'depths', 'focal_levels',
                'focal_windows', 'use_overlapped_embed', 'use_postln',
                'use_layer_scale', 'normalize_modulator',
                'use_postln_in_modulation'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_layers = len(self.depths)

        self.out_indices = out_indices
        self.out_after_downsample = out_after_downsample
        self.frozen_stages = frozen_stages

        if self.arch_settings['use_overlapped_embed']:
            _patch_cfg = dict(
                in_channels=in_channels,
                input_size=None,
                embed_dims=self.embed_dims,
                conv_type='Conv2d',
                kernel_size=7,
                padding=3,
                stride=4,
                norm_cfg=dict(type='LN'),
            )
            _patch_cfg.update(patch_cfg)
            self.patch_embed = OverlappedPatchEmbed(**_patch_cfg)
        else:
            _patch_cfg = dict(
                in_channels=in_channels,
                input_size=None,
                embed_dims=self.embed_dims,
                conv_type='Conv2d',
                kernel_size=patch_size,
                stride=patch_size,
                norm_cfg=dict(type='LN'),
            )
            _patch_cfg.update(patch_cfg)
            self.patch_embed = PatchEmbed(**_patch_cfg)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.layers = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth, focal_level, focal_window) in enumerate(
                zip(self.depths, self.arch_settings['focal_levels'],
                    self.arch_settings['focal_windows'])):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {
                'embed_dims':
                int(self.embed_dims * 2**i),
                'depth':
                depth,
                'ffn_ratio':
                self.arch_settings['ffn_ratio'],
                'drop_rate':
                drop_rate,
                'drop_paths':
                dpr[:depth],
                'norm_cfg':
                norm_cfg,
                'downsample':
                downsample,
                'focal_level':
                focal_level,
                'focal_window':
                focal_window,
                'use_overlapped_embed':
                self.arch_settings['use_overlapped_embed'],
                'use_layer_scale':
                self.arch_settings['use_layer_scale'],
                'use_postln':
                self.arch_settings['use_postln'],
                'normalize_modulator':
                self.arch_settings['normalize_modulator'],
                'use_postln_in_modulation':
                self.arch_settings['use_postln_in_modulation'],
                'with_cp':
                with_cp,
                **stage_cfg
            }
            layer = BasicLayer(**_stage_cfg)
            self.layers.append(layer)

            dpr = dpr[depth:]
            embed_dims.append(layer.out_channels)

        if self.out_after_downsample:
            self.num_features = embed_dims[1:]
        else:
            self.num_features = embed_dims[:-1]

        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg,
                                              self.num_features[i])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm{i}', norm_layer)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

            self.drop_after_pos.eval()

        for i in range(0, self.frozen_stages + 1):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for i in self.out_indices:
            if i <= self.frozen_stages:
                for param in getattr(self, f'norm{i}').parameters():
                    param.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None or 'checkpoint' not in self.init_cfg:
            super().init_weights()
        else:
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            prefix = self.init_cfg.get('prefix', None)
            if prefix is not None:
                if not prefix.endswith('.'):
                    prefix += '.'
                prefix_len = len(prefix)

                state_dict = {
                    k[prefix_len:]: v
                    for k, v in state_dict.items() if k.startswith(prefix)
                }

                assert state_dict, f'{prefix} is not in the pretrained model'

            focal_layers_keys = [
                k for k in state_dict.keys()
                if ('focal_layers' in k and 'bias' not in k)
            ]
            for table_key in focal_layers_keys:
                if table_key not in self.state_dict():
                    continue
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]

                if len(table_pretrained.shape) != 4:
                    L1 = table_pretrained.shape[1]
                    L2 = table_current.shape[1]

                    if L1 != L2:
                        S1 = int(L1**0.5)
                        S2 = int(L2**0.5)
                        table_pretrained_resized = F.interpolate(
                            table_pretrained.view(1, 1, S1, S1),
                            size=(S2, S2),
                            mode='bicubic')
                        state_dict[table_key] = table_pretrained_resized.view(
                            1, L2) * L1 / L2
                else:
                    fsize1 = table_pretrained.shape[2]
                    fsize2 = table_current.shape[2]

                    # NOTE: different from interpolation used in
                    # self-attention, we use padding or clipping for focal conv
                    if fsize1 < fsize2:
                        table_pretrained_resized = torch.zeros(
                            table_current.shape)
                        table_pretrained_resized[:, :, (fsize2 - fsize1) //
                                                 2:-(fsize2 - fsize1) // 2,
                                                 (fsize2 - fsize1) //
                                                 2:-(fsize2 - fsize1) //
                                                 2] = table_pretrained
                        state_dict[table_key] = table_pretrained_resized
                    elif fsize1 > fsize2:
                        table_pretrained_resized = table_pretrained[:, :, (
                            fsize1 - fsize2) // 2:-(fsize1 - fsize2) // 2, (
                                fsize1 - fsize2) // 2:-(fsize1 - fsize2) // 2]
                        state_dict[table_key] = table_pretrained_resized

            f_layers_keys = [
                k for k in state_dict.keys() if ('modulation.f' in k)
            ]
            for table_key in f_layers_keys:
                if table_key not in self.state_dict():
                    continue
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                if table_pretrained.shape != table_current.shape:
                    if len(table_pretrained.shape) == 2:
                        # for linear weights
                        dim = table_pretrained.shape[1]
                        assert table_current.shape[1] == dim
                        L1 = table_pretrained.shape[0]
                        L2 = table_current.shape[0]

                        if L1 < L2:
                            table_pretrained_resized = torch.zeros(
                                table_current.shape)
                            # copy for linear project
                            (table_pretrained_resized[:2 * dim]
                             ) = table_pretrained[:2 * dim]
                            # copy for global token gating
                            table_pretrained_resized[-1] = table_pretrained[-1]
                            # copy for first multiple focal levels
                            table_pretrained_resized[2 * dim:2 * dim + (
                                L1 - 2 * dim - 1)] = table_pretrained[2 *
                                                                      dim:-1]
                            # reassign pretrained weights
                            state_dict[table_key] = table_pretrained_resized
                        elif L1 > L2:
                            raise NotImplementedError
                    elif len(table_pretrained.shape) == 1:
                        # for linear bias
                        L1 = table_pretrained.shape[0]
                        L2 = table_current.shape[0]
                        if L1 < L2:
                            table_pretrained_resized = torch.zeros(
                                table_current.shape)
                            # copy for linear project
                            (table_pretrained_resized[:2 * dim]
                             ) = table_pretrained[:2 * dim]
                            # copy for global token gating
                            table_pretrained_resized[-1] = table_pretrained[-1]
                            # copy for first multiple focal levels
                            table_pretrained_resized[2 * dim:2 * dim + (
                                L1 - 2 * dim - 1)] = table_pretrained[2 *
                                                                      dim:-1]
                            # reassign pretrained weights
                            state_dict[table_key] = table_pretrained_resized
                        elif L1 > L2:
                            raise NotImplementedError

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        """Forward function."""
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer(
                x, hw_shape, do_downsample=self.out_after_downsample)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            if layer.downsample is not None and not self.out_after_downsample:
                x = x.transpose(1, 2).reshape(x.shape[0], -1, *hw_shape)
                x, hw_shape = layer.downsample(x)

        return tuple(outs)

    def train(self, mode=True):
        super(FocalNet, self).train(mode)
        self._freeze_stages()

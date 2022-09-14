# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.registry import MODELS
from torch import nn

from .base_backbone import BaseBackbone
from .mobilenet_v2 import InvertedResidual
from .vision_transformer import TransformerEncoderLayer


class MobileVitBlock(nn.Module):
    """ MobileViT block
        Paper: https://arxiv.org/abs/2110.02178?context=cs.LG
    """

    def __init__(
            self,
            in_channels: int,
            transformer_dim,
            ffn_dim,
            out_channels,
            conv_ksize: int = 3,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish'),
            num_transformer_blocks: int = 2,
            patch_size: int = 8,
            num_heads: int = 4,
            drop_rate: float = 0.,
            attn_drop_rate=0.,
            drop_path_rate: int = 0.,
            no_fusion: bool = False,
            transformer_norm_cfg: Callable = dict(type='LN'),
            **kwargs,  # eat unused args
    ):
        super(MobileVitBlock, self).__init__()

        self.local_rep = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=conv_ksize,
                padding=int((conv_ksize - 1) / 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=in_channels,
                out_channels=transformer_dim,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None),
        )

        global_rep = [
            TransformerEncoderLayer(
                embed_dims=transformer_dim,
                num_heads=num_heads,
                feedforward_channels=ffn_dim,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                qkv_bias=True,
                act_cfg=dict(type='Swish'),
                norm_cfg=transformer_norm_cfg)
            for _ in range(num_transformer_blocks)
        ]
        global_rep.append(
            build_norm_layer(transformer_norm_cfg, transformer_dim)[1])
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = ConvModule(
            in_channels=transformer_dim,
            out_channels=out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = ConvModule(
                in_channels=in_channels + out_channels,
                out_channels=out_channels,
                kernel_size=conv_ksize,
                padding=int((conv_ksize - 1) / 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        self.patch_size = (patch_size, patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Local representation
        x = self.local_rep(x)

        # Unfold (feature map -> patches)
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(
            W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w # noqa
        num_patches = num_patch_h * num_patch_w  # N
        interpolate = False
        if new_h != H or new_w != W:
            # Note: Padding can be done, but then it needs to be handled in attention function. # noqa
            x = F.interpolate(
                x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            interpolate = True

        # [B, C, H, W] --> [B * C * n_h, n_w, p_h, p_w]
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w,
                      patch_w).transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [BP, N, C] where P = p_h * p_w and N = n_h * n_w # noqa
        x = x.reshape(B, C, num_patches,
                      self.patch_area).transpose(1, 3).reshape(
                          B * self.patch_area, num_patches, -1)

        # Global representations
        x = self.global_rep(x)

        # Fold (patch -> feature map)
        # [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w,
                                      patch_h, patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W] # noqa
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h,
                                      num_patch_w * patch_w)
        if interpolate:
            x = F.interpolate(
                x, size=(H, W), mode='bilinear', align_corners=False)

        x = self.conv_proj(x)
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        return x


@MODELS.register_module()
class MobileViT(BaseBackbone):
    """MobileViT backbone.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    # Parameters to build layers. The first param is the type of layer.
    # For `mobilenetv2` layer, the rest params from left to right are:
    #     out channels, stride, num of blocks, expand_ratio.
    # For `mobilevit` layer, the rest params from left to right are:
    #     out channels, transformer_channels, ffn channels,
    # stride, num of transformer blocks, expand_ratio.
    arch_settings = {
        'small': [['mobilenetv2', 32, 1, 1, 4], ['mobilenetv2', 64, 2, 3, 4],
                  ['mobilevit', 96, 2, 144, 288, 2, 4],
                  ['mobilevit', 128, 2, 192, 384, 4, 4],
                  ['mobilevit', 160, 2, 240, 480, 3, 4]]
    }

    def __init__(self,
                 arch='small',
                 in_channels=3,
                 stem_channels=16,
                 last_exp_factor=4,
                 out_indices=(5, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Swish'),
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(MobileViT, self).__init__(init_cfg)
        arch_settings = self.arch_settings[arch]
        self.num_stages = len(arch_settings)

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

        _make_layer_func = {
            'mobilenetv2': self._make_mobilenetv2_layer,
            'mobilevit': self._make_mobilevit_layer,
        }

        self.stem = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        in_channels = stem_channels
        layers = []
        for i, layer_settings in enumerate(arch_settings):
            layer_type, settings = layer_settings[0], layer_settings[1:]
            layer, out_channels = _make_layer_func[layer_type](in_channels,
                                                               *settings)
            layers.append(layer)
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

        self.conv_1x1_exp = ConvModule(
            in_channels=in_channels,
            out_channels=last_exp_factor * in_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    @staticmethod
    def _make_mobilevit_layer(in_channels,
                              out_channels,
                              stride,
                              transformer_dim,
                              ffn_dim,
                              num_transformer_blocks,
                              expand_ratio=4):
        layer = []
        layer.append(
            InvertedResidual(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            ))
        layer.append(
            MobileVitBlock(
                in_channels=out_channels,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                out_channels=out_channels,
                num_transformer_blocks=num_transformer_blocks,
            ))
        return nn.Sequential(*layer), out_channels

    @staticmethod
    def _make_mobilenetv2_layer(in_channels,
                                out_channels,
                                stride,
                                num_blocks,
                                expand_ratio=4):

        layer = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1

            layer.append(
                InvertedResidual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expand_ratio=expand_ratio,
                ))
            in_channels = out_channels
        return nn.Sequential(*layer), out_channels

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

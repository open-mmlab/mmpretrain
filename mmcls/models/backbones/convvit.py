# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.cnn.bricks import DropPath, build_norm_layer, build_activation_layer
from mmcv.runner.base_module import BaseModule

from ..utils import to_2tuple
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .vision_transformer import TransformerEncoderLayer


class ConvMlp(BaseModule):
    """Implements the Mlp part for ConvBlock. The main difference is ConvMlp
    implies N, C, H, W input format.

    Args:
        embed_dims (int): the feature dimension
        feedforward_channels (int): the MLP hidden feature dimension
        drop_rate (float): probability of an element to be zeroed between
            two fully-connected layers. defaults to 0.
        act_cfg (dict): config for the activation between the two 
            fully-connected layers. defaults to ``dict(type='GELU')``.
        init_cfg (dict, optional): initialization config dict. defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 act_cfg: dict = dict(type='GELU'),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.fc1 = nn.Conv2d(embed_dims, feedforward_channels, 1)
        self.act = build_activation_layer(act_cfg)
        self.dropout = (
            nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity())
        self.fc2 = nn.Conv2d(feedforward_channels, embed_dims, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class ConvBlock(BaseModule):
    """Implements one convolutional block in ConvViT.

    Args:
        embed_dims (int): the feature dimension
        feedforward_channels (int): the MLP hidden feature dimension
        drop_rate (float): probability of an element to be zeroed
            in the feed-forward layer. default is 0.
        act_cfg (dict): the activation config for the MLP.
            defaults to ``dict(type='GELU')``
        norm_cfg (dict): the normalization config.
            defaults to ``dict(type='LN2d')``
        init_cfg (dict, optional): initialization config dict.
    """

    def __init__(self,
                 embed_dims: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg=None):
        super(ConvBlock, self).__init__(init_cfg=init_cfg)

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.conv1 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.conv2 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.attn = nn.Conv2d(embed_dims, embed_dims,
            kernel_size=5,
            padding=2,
            groups=embed_dims)

        self.drop_path = \
            DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.mlp = ConvMlp(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            drop_rate=drop_rate,
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        x_id = x

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)

        x = x_id + self.drop_path(x)
        x_id = x

        x = self.norm2(x)
        x = self.mlp(x)
        
        x = x_id + self.drop_path(x)
        x_id = x

        return x

@BACKBONES.register_module()
class ConvViT(BaseBackbone):
    """ConvViT, a mixed convolution-transformer architecture.
    
    An implementation of  ConvViT architecture, as used in MCMAE (previously 
    known as ConvMAE, https://arxiv.org/abs/2205.03892)

    Args:
        arch (str | dict): ConvViT architecture. If use string, choose from
            'base'. If use dict, it should have the following keys:

            - **embed_dims** (list[int]): the dimensions of embeddings,
              one int for each stage.
            - **num_layers** (list[int]): number of transformer encoder layers
              in each stage.
            - **num_heads** int: number of heads in the self-attention layers
            - **feedforward_channels** (list[int]): the hidden dimensions in
              feedforward modules.

            defaults to 'base'.
        img_size (int | tuple[int]): the expected input image shape. defaults
            to 224.
        patch_size (int | tuple[int]): the patch size used to downsample spatial
            size before each stage. defaults to [4, 2, 2].
        in_channels (int): the number of input channels. defaults to 3.
        out_indices (int | Sequence[int]): indices for blocks of whose outputs
            the final output consists of. defaults to -1 (last block only).
        drop_rate (float): probability of an element to be zeroed.
            defaults to 0.
        drop_path_rate (float): probability of a layer to be zeroed.
            defaults to 0. note that the actual drop path rate for each block
            increases linearly with the depth of the block.
        qkv_bias (bool): whether to add bias for qkv in attentio nmodules.
            defaults to true. convolutional blocks are not affected.
        tx_norm_cfg (dict): transformer block norm config. defaults to
            ``dict(type='LN', eps=1e-6)``
        conv_norm_cfg (dict): convolutional block norm config. defaults to 
            ``dict(type='LN2d', eps=1e-6)``
        final_norm (bool): whether to add an additional layer to normalize the
            final feature map. defaults to true. the final norm will use
            ``tx_norm_cfg``.
        interpolate_mode (str): select the interpolation mode for position
            embeddings vector resize. defaults to 'bicubic'.
        patch_cfg (dict): configs for patch embeddings. defaults to an empty
            dict.
        layer_cfgs (dict | Sequence[dict]): configs for each transformer or
            convolutional layer. defaults to an empty dict.
        act_cfg (dict): configs for the activation function. default is 
            ``dict(type='GELU')``.
        init_cfg (dict, optional): initialization config dict. defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': [256, 384, 768],
                'num_layers': [2, 2, 11],
                'num_heads': 12,
                'feedforward_channels': [256 * 4, 384 * 4, 768 * 4],
            }),
    }

    def __init__(self,
                 arch: Union[str, dict] = 'base',
                 img_size: Union[int, Tuple[int]] = 224,
                 patch_size: Sequence[int] = [4, 2, 2],
                 in_channels: int = 3,
                 out_indices: Union[int, Sequence[int]] = -1,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 tx_norm_cfg: dict = dict(type='LN', eps=1e-6),
                 conv_norm_cfg: dict = dict(type='LN2d', eps=1e-6),
                 final_norm: bool = True,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 layer_cfgs: Union[dict, Sequence[dict]] = dict(),
                 act_cfg: dict = dict(type='GELU'),
                 init_cfg: Optional[dict] = None):
        super(ConvViT, self).__init__(init_cfg)

        essential_keys = {
            'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels',
        }
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in self.arch_zoo, \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        img_size = to_2tuple(img_size)
        embed_dims = self.arch_settings['embed_dims']
        num_layers = self.arch_settings['num_layers']
        num_heads = self.arch_settings['num_heads']
        feedforward_channels = self.arch_settings['feedforward_channels']

        num_stages = 3
        assert num_stages == len(embed_dims), \
            'embed_dims has to be specified for each stage exactly once.'
        assert num_stages == len(patch_size), \
            'patch_size has to be specified for each stage exactly once.'
        assert num_stages == len(num_layers), \
            'num_layers has to be specified for each stage exactly once.'
        assert num_stages == len(feedforward_channels), \
            'feedforward_channels has to be specified for each stage ' \
            'exactly once.'

        dpr = np.linspace(0, drop_path_rate, sum(num_layers))

        self.patch_embeds = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        self.patch_embed_act = build_activation_layer(act_cfg)

        stage_img_size = img_size
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * sum(num_layers)
        for stage_idx in range(num_stages):
            _patch_cfg = dict(
                in_channels=(
                    in_channels if stage_idx == 0 else embed_dims[stage_idx - 1]
                ),
                input_size=stage_img_size,
                embed_dims=embed_dims[stage_idx],
                conv_type='Conv2d',
                kernel_size=patch_size[stage_idx],
                stride=patch_size[stage_idx],
                norm_cfg=tx_norm_cfg,
            )
            _patch_cfg.update(patch_cfg)
            patch_embed = PatchEmbed(**_patch_cfg)
            self.patch_embeds.append(patch_embed)
            stage_img_size = patch_embed.init_out_size

            if stage_idx == 2:
                self.patch_embeds.append(
                    nn.Linear(embed_dims[stage_idx], embed_dims[stage_idx]))


            blocks = []
            for layer_idx in range(num_layers[stage_idx]):
                global_layer_idx = sum(num_layers[:stage_idx]) + layer_idx
                _layer_cfg = dict(
                    embed_dims=embed_dims[stage_idx],
                    feedforward_channels=feedforward_channels[stage_idx],
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[global_layer_idx],
                    act_cfg=act_cfg)
                if stage_idx >= 2: # transformer stage
                    layer_class = TransformerEncoderLayer
                    _layer_cfg.update(
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        norm_cfg=tx_norm_cfg,
                    )
                    self.pos_embed = nn.Parameter(
                        torch.zeros(1, *stage_img_size, embed_dims[stage_idx]))
                else: # conv stage
                    layer_class = ConvBlock
                    layer_conv_norm_cfg = conv_norm_cfg.copy()
                    _layer_cfg.update(
                        norm_cfg=layer_conv_norm_cfg,
                        )
                _layer_cfg.update(layer_cfgs[global_layer_idx])
                blocks.append(layer_class(**_layer_cfg))
            self.blocks.append(nn.ModuleList(blocks))
        
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                tx_norm_cfg, embed_dims[-1], postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.interpolate_mode = interpolate_mode
        self.out_indices = (
                [out_indices] if isinstance(out_indices, int) else out_indices)
        self.out_indices = [
                x if x >= 0 else sum(num_layers) + x for x in self.out_indices]
        self.drop_after_pos = (
            nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity())

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super().init_weights()
        if not (isinstance(self.init_cfg, dict) and \
                self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        num_stages = len(self.blocks)
        name = prefix + 'pos_embed'
        if name not in state_dict:
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmcv.utils import print_log
            logger = get_root_logger()
            print_log(
                f'Resize pos_embed.{i} shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embeds[i].shape}.',
                logger=logger)
            state_dict[name] = resize_pos_embed(state_dict[name],
                ckpt_pos_embed_shape[1:-1],
                self.pos_embeds[i].shape[1:-1],
                self.interpolate_mode, 0)

    def forward(self, x):
        outs = []

        for stage_idx in range(len(self.blocks)):
            stage_type = 'c' if stage_idx < 2 else 't'

            x, stage_out_size = self.patch_embeds[stage_idx](x)
            x = self.patch_embed_act(x)
            def convert_nlc_to_nchw(x):
                x = x.view(x.size(0), *stage_out_size, x.size(-1))
                x = x.permute(0, -1, *range(1, x.ndim - 1)).contiguous()
                return x

            if stage_type == 't':
                x = self.patch_embeds[3](x)
                x = x + self.pos_embed.flatten(1, -2)
                x = self.drop_after_pos(x)
            elif stage_type  == 'c':
                x = convert_nlc_to_nchw(x)
            else:
                raise NotImplementedError()

            for layer_idx, block in enumerate(self.blocks[stage_idx]):
                global_layer_idx = sum(
                    [len(x) for x in self.blocks[:stage_idx]]) + layer_idx
                x = block(x)
                if global_layer_idx == sum([len(x) for x in self.blocks]) - 1 \
                        and self.final_norm:
                    x = self.norm1(x)
                if global_layer_idx in self.out_indices:                  
                    outs.append({
                        't': convert_nlc_to_nchw,
                        'c': lambda x: x,
                        }[stage_type](x))

            if stage_type == 't':
                x = convert_nlc_to_nchw(x)

        return tuple(outs)

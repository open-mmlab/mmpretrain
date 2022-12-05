# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList

from mmcls.registry import MODELS
from ..utils import (BEiTAttention, resize_pos_embed,
                     resize_relative_position_bias_table, to_2tuple)
from .vision_transformer import TransformerEncoderLayer, VisionTransformer


class RelativePositionBias(BaseModule):
    """Relative Position Bias.

    This module is copied from
    https://github.com/microsoft/unilm/blob/master/beit/modeling_finetune.py#L209.

    Args:
        window_size (Sequence[int]): The window size of the relative
            position bias.
        num_heads (int): The number of head in multi-head attention.
        with_cls_token (bool): To indicate the backbone has cls_token or not.
            Defaults to True.
    """

    def __init__(
        self,
        window_size: Sequence[int],
        num_heads: int,
        with_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        if with_cls_token:
            num_extra_tokens = 3
        else:
            num_extra_tokens = 0
        # cls to token & token to cls & cls to cls
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1) + num_extra_tokens
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance,
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each
        # token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] -\
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        if with_cls_token:
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1, ) * 2,
                dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(
                -1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
        else:
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1], ) * 2,
                dtype=relative_coords.dtype)
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer('relative_position_index',
                             relative_position_index)

    def forward(self) -> torch.Tensor:
        # Wh*Ww,Wh*Ww,nH
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)
        return relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class BEiTTransformerEncoderLayer(TransformerEncoderLayer):
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
            the learnable scaling of attention and FFN. 1 means no scaling.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        window_size (tuple[int]): The height and width of the window.
            Defaults to None.
        use_rel_pos_bias (bool): Whether to use unique relative position bias,
            if False, use shared relative position bias defined in backbone.
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
                 window_size: Tuple[int, int],
                 use_rel_pos_bias: bool,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 bias: Union[str, bool] = 'qv_bias',
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 attn_cfg: dict = dict(),
                 ffn_cfg: dict = dict(add_identity=False),
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
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

        attn_cfg = {
            'window_size': window_size,
            'use_rel_pos_bias': use_rel_pos_bias,
            'qk_scale': None,
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'attn_drop': attn_drop_rate,
            'proj_drop': drop_rate,
            'bias': bias,
            **attn_cfg,
        }
        self.attn = BEiTAttention(**attn_cfg)

        ffn_cfg = {
            'embed_dims': embed_dims,
            'feedforward_channels': feedforward_channels,
            'num_fcs': num_fcs,
            'ffn_drop': drop_rate,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path_rate),
            'act_cfg': act_cfg,
            **ffn_cfg,
        }
        self.ffn = FFN(**ffn_cfg)

        # NOTE: drop path for stochastic depth, we shall see if
        # this is better than dropout here
        dropout_layer = dict(type='DropPath', drop_prob=drop_path_rate)
        self.drop_path = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

        if layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dims)),
                requires_grad=True)
            self.gamma_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dims)),
                requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor,
                rel_pos_bias: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(
                self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        return x


@MODELS.register_module()
class BEiT(VisionTransformer):
    """Backbone for BEiT.

    A PyTorch implement of : `BEiT: BERT Pre-Training of Image Transformers
    <https://arxiv.org/abs/2106.08254>`_
    A PyTorch implement of : `BEiT v2: Masked Image Modeling with
    Vector-Quantized Visual Tokenizers <https://arxiv.org/abs/2208.06366>`_

    Args:
        arch (str | dict): BEiT architecture. If use string, choose from
        'base', 'large'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        avg_token (bool): Whether or not to use the mean patch token for
            classification. If True, the model will only take the average
            of all patch tokens. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        use_abs_pos_emb (bool): Use position embedding like vanilla ViT.
            Defaults to False.
        use_rel_pos_bias (bool): Use relative position embedding in each
            transformer encoder layer. Defaults to True.
        use_shared_rel_pos_bias (bool): Use shared relative position embedding,
            all transformer encoder layers share the same relative position
            embedding. Defaults to False.
        layer_scale_init_value (float): The initialization value for
            the learnable scaling of attention and FFN. Defaults to 0.1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=False,
                 with_cls_token=True,
                 avg_token=True,
                 frozen_stages=-1,
                 output_cls_token=False,
                 use_abs_pos_emb=False,
                 use_rel_pos_bias=True,
                 use_shared_rel_pos_bias=False,
                 layer_scale_init_value=0.1,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(VisionTransformer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        self.interpolate_mode = interpolate_mode

        # Set position embedding
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.num_extra_tokens,
                            self.embed_dims))
            self._register_load_state_dict_pre_hook(self._prepare_pos_embed)
        else:
            self.pos_embed = None
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        assert not (use_rel_pos_bias and use_shared_rel_pos_bias), (
            '`use_rel_pos_bias` and `use_shared_rel_pos_bias` cannot be set '
            'to True at the same time')
        self.use_rel_pos_bias = use_rel_pos_bias

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_resolution,
                num_heads=self.arch_settings['num_heads'])
        else:
            self.rel_pos_bias = None
        self._register_load_state_dict_pre_hook(
            self._prepare_relative_position_bias_table)

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
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                layer_scale_init_value=layer_scale_init_value,
                window_size=self.patch_resolution,
                use_rel_pos_bias=use_rel_pos_bias,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(BEiTTransformerEncoderLayer(**_layer_cfg))

        self.frozen_stages = frozen_stages
        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.avg_token = avg_token
        if avg_token:
            self.norm2_name, norm2 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=2)
            self.add_module(self.norm2_name, norm2)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        rel_pos_bias = self.rel_pos_bias() \
            if self.rel_pos_bias is not None else None

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, rel_pos_bias)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                if self.with_cls_token:
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = x[:, 0]
                else:
                    patch_token = x.reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None

                if self.avg_token:
                    patch_token = patch_token.permute(0, 2, 3, 1)
                    patch_token = patch_token.reshape(
                        B, patch_resolution[0] * patch_resolution[1],
                        C).mean(dim=1)
                    patch_token = self.norm2(patch_token)
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)

        return tuple(outs)

    def _prepare_relative_position_bias_table(self, state_dict, prefix, *args,
                                              **kwargs):
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()

        if self.use_rel_pos_bias and 'rel_pos_bias.relative_position_bias_table' in state_dict:  # noqa:E501
            logger.info('Expand the shared relative position embedding to '
                        'each transformer block.')
            rel_pos_bias = state_dict[
                'rel_pos_bias.relative_position_bias_table']
            for i in range(self.num_layers):
                state_dict[
                    f'layers.{i}.attn.relative_position_bias_table'] = \
                        rel_pos_bias.clone()
            state_dict.pop('rel_pos_bias.relative_position_bias_table')
            state_dict.pop('rel_pos_bias.relative_position_index')

        state_dict_model = self.state_dict()
        all_keys = list(state_dict_model.keys())
        for key in all_keys:
            if 'relative_position_bias_table' in key:
                ckpt_key = prefix + key
                if ckpt_key not in state_dict:
                    continue
                rel_pos_bias_pretrained = state_dict[ckpt_key]
                rel_pos_bias_current = state_dict_model[key]
                L1, nH1 = rel_pos_bias_pretrained.size()
                L2, nH2 = rel_pos_bias_current.size()
                src_size = int((L1 - 3)**0.5)
                dst_size = int((L2 - 3)**0.5)
                if L1 != L2:
                    extra_tokens = rel_pos_bias_pretrained[-3:, :]
                    rel_pos_bias = rel_pos_bias_pretrained[:-3, :]

                    new_rel_pos_bias = resize_relative_position_bias_table(
                        src_size, dst_size, rel_pos_bias, nH1)
                    new_rel_pos_bias = torch.cat(
                        (new_rel_pos_bias, extra_tokens), dim=0)
                    logger.info('Resize the relative_position_bias_table from '
                                f'{state_dict[ckpt_key].shape} to '
                                f'{new_rel_pos_bias.shape}')
                    state_dict[ckpt_key] = new_rel_pos_bias

                    # The index buffer need to be re-generated.
                    index_buffer = ckpt_key.replace('bias_table', 'index')
                    if index_buffer in state_dict:
                        del state_dict[index_buffer]

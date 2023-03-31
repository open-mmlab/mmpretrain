# Copyright (c) OpenMMLab. All rights reserved.
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, ModuleList

from mmpretrain.registry import MODELS
from ..utils import RotaryEmbeddingFast, build_norm_layer, resize_pos_embed
from .vision_transformer import VisionTransformer


class SwiGLUFFN(BaseModule):
=======
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from ..utils import build_norm_layer, resize_pos_embed, to_2tuple


class RotaryEmbedding(BaseModule):
    """Implements 2D rotary embedding (RoPE) for image tokens.

    Position encoding is implemented with sin and cos functions,

        .. math::
            Pos_{cos} = cos(\frac{t}{\theta^{\frac{2i}{d}}} \\
            Pos_{sin} = sin(\frac{t}{\theta^{\frac{2i}{d}}}

    Args:
        embed_dims (int): The feature dimension for each head.
        patch_resolution (int | tuple): The resolution of the
            image, in format (H, W).
        theta (float): The hyperparameter for position coding.
            Defaults to 10000.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 patch_resolution,
                 theta=10000.,
                 init_cfg=None):
        super(RotaryEmbedding, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.patch_resolution = to_2tuple(patch_resolution)
        self.theta = theta

        freqs_cos, freqs_sin = self.compute_position_embedding()
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

    def compute_position_embedding(self):
        frequency = self.theta**(
            torch.arange(0, self.embed_dims, 2).float() / self.embed_dims)
        frequency = 1. / frequency
        half_dim = frequency.shape[0]

        h, w = self.patch_resolution
        th = torch.arange(h) / h * half_dim
        tw = torch.arange(w) / w * half_dim

        position_h = (th[:, None] @ frequency[None, :]).repeat(1, 2)
        position_w = (tw[:, None] @ frequency[None, :]).repeat(1, 2)

        height = position_h[:, None, :].expand(h, w, self.embed_dims)
        width = position_w[None, :, :].expand(h, w, self.embed_dims)
        position = torch.cat((height, width), dim=-1)

        freqs_cos = position.cos().view(-1, position.shape[-1])
        freqs_sin = position.sin().view(-1, position.shape[-1])

        return freqs_cos, freqs_sin

    def forward(self, x, patch_resolution):
        # Check whether the patch resolution is the predefined size
        patch_resolution = to_2tuple(patch_resolution)
        if patch_resolution != self.patch_resolution:
            self.patch_resolution = patch_resolution
            freqs_cos, freqs_sin = self.compute_position_embedding()
            self.register_buffer('freqs_cos', freqs_cos.to(x.device))
            self.register_buffer('freqs_sin', freqs_sin.to(x.device))

        batch, num_heads, num_patches, dim = x.shape

        inputs = x
        x = x.reshape(batch, num_heads, num_patches, -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = x.reshape(batch, num_heads, num_patches, dim)

        return inputs * self.freqs_cos + x * self.freqs_sin


class SwiGLU(BaseModule):
>>>>>>> feat: add eva02 backbone
    """Implements SwiGLU with sub layer norm.

    Args:
        in_features (int): The feature dimension.
        hidden_features (int, optional): The hidden dimension of features,
            and if None, use ``in_features``. Defaults to None.
        out_features (int, optional): The dimension of outputs, and if None,
            use ``in_features``. Defaults to None.
        norm_cfg (dict, optional): Config dict for sub normalization layer.
            Defaults to ``dict(type='LN')``.
        act_cfg (dict): The activation config.
            Defaults to ``dict(type='SiLU')``.
        drop_rate (float): The dropout rate. Defaults to 0.
        sub_ln (bool): Whether to add the sub layer normalization.
            Defaults to False.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='SiLU'),
                 drop_rate=0.,
                 sub_ln=False,
                 init_cfg=None):
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
        super(SwiGLUFFN, self).__init__(init_cfg=init_cfg)
=======
        super(SwiGLU, self).__init__(init_cfg=init_cfg)
>>>>>>> feat: add eva02 backbone

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        if sub_ln:
            self.norm = build_norm_layer(norm_cfg, hidden_features)
        else:
            self.norm = nn.Identity()

        self.act = build_activation_layer(act_cfg)

        self.w3 = nn.Linear(hidden_features, out_features)

        self.dropout_layer = nn.Dropout(drop_rate)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.norm(hidden)
        x = self.w3(x)
        x = self.dropout_layer(x)
        return x


class AttentionWithRoPE(BaseModule):
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
    """Multi-head Attention Module with 2D sincos position embedding (RoPE).
=======
    """Multi-head Attention Module with 2D rotary position embedding (RoPE).
>>>>>>> feat: add eva02 backbone

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        qkv_bias (bool): If True, add a learnable bias to q and v. Note
            that we follows the official implementation where ``k_bias``
            is 0. Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        rope (:obj:`torch.nn.Module`, optional): If it is an object of the
            ``RotaryEmbedding``, the rotation of the token position will be
            performed before the softmax. Defaults to None.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 rope=None,
                 with_cls_token=True,
                 init_cfg=None):
        super(AttentionWithRoPE, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
=======

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.v_bias = None
>>>>>>> feat: add eva02 backbone

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.with_cls_token = with_cls_token
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536

=======
>>>>>>> feat: add eva02 backbone
        self.rope = rope

    def forward(self, x, patch_resolution):
        B, N, _ = x.shape

<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
        qkv = self.qkv(x)
=======
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        else:
            qkv_bias = None

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
>>>>>>> feat: add eva02 backbone
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        if self.rope:
            if self.with_cls_token:
                q_t = q[:, :, 1:, :]
                ro_q_t = self.rope(q_t, patch_resolution)
                q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

                k_t = k[:, :, 1:, :] if self.with_cls_token else k
                ro_k_t = self.rope(k_t, patch_resolution)
                k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)
            else:
                q = self.rope(q, patch_resolution)
                k = self.rope(k, patch_resolution)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
class EVA02EndcoderLayer(BaseModule):
    """Implements one encoder EVA02EndcoderLayer in EVA02.
=======
class Block(BaseModule):
    """Implements one encoder block in EVA02.
>>>>>>> feat: add eva02 backbone

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
        feedforward_channels (int): The hidden dimension of FFNs.
=======
        mlp_ratio (float): The ratio of the mlp module.
            Defaults to 4*2/3.
>>>>>>> feat: add eva02 backbone
        sub_ln (bool): Whether to add the sub layer normalization
            in the attention module. Defaults to False.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool): enable bias for projection in the attention module
            if True. Defaults to True.
        rope (:obj:`torch.nn.Module`, optional): RotaryEmbedding object
            in the attention module. Defaults to None.
        drop_rate (float): Dropout rate in the mlp module. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
                 feedforward_channels,
=======
                 mlp_ratio=4 * 2 / 3.,
>>>>>>> feat: add eva02 backbone
                 sub_ln=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=False,
                 qk_scale=None,
                 proj_bias=True,
                 rope=None,
                 with_cls_token=True,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
        super(EVA02EndcoderLayer, self).__init__(init_cfg=init_cfg)
=======
        super(Block, self).__init__(init_cfg=init_cfg)
>>>>>>> feat: add eva02 backbone

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)

        self.attn = AttentionWithRoPE(
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
=======
            rope=rope,
>>>>>>> feat: add eva02 backbone
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_bias=proj_bias,
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
            rope=rope,
=======
>>>>>>> feat: add eva02 backbone
            with_cls_token=with_cls_token)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)

<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
        self.mlp = SwiGLUFFN(
            in_features=embed_dims,
            hidden_features=feedforward_channels,
=======
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = SwiGLU(
            in_features=embed_dims,
            hidden_features=mlp_hidden_dim,
>>>>>>> feat: add eva02 backbone
            sub_ln=sub_ln,
            drop_rate=drop_rate,
            norm_cfg=norm_cfg,
        )

    def forward(self, x, patch_resolution):
        inputs = x
        x = self.norm1(x)
        x = self.attn(x, patch_resolution)
        x = self.drop_path(x)
        x = inputs + x

        inputs = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = inputs + x

        return x


@MODELS.register_module()
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
class EVA02(VisionTransformer):
=======
class EVA02(BaseModule):
>>>>>>> feat: add eva02 backbone
    """EVA02 Vision Transformer.

    A PyTorch implement of : `EVA-02: A Visual Representation for Neon Genesis
    <https://arxiv.org/abs/2303.11331>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'tiny', 'small', 'base', 'large'. If use dict,
            it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **mlp_ratio** (float): The ratio of the mlp module.

            Defaults to 'tiny'.
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536

        sub_ln (bool): Whether to add the sub layer normalization in swiglu.
            Defaults to False.
=======
        img_size (int | tuple): The expected input image shape.
            Defaults to 336.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 14.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
>>>>>>> feat: add eva02 backbone
        drop_rate (float): Probability of an element to be zeroed in the
            mlp module. Defaults to 0.
        attn_drop_rate (float): Probability of an element to be zeroed after
            the softmax in the attention. Defaults to 0.
        proj_drop_rate (float): Probability of an element to be zeroed after
            projection in the attention. Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        **kwargs(dict, optional): Other args for Vision Transformer.
=======
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            Defaults to ``"cls_token"``.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        sub_ln (bool): Whether to add the sub layer normalization in swiglu.
            Defaults to False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
>>>>>>> feat: add eva02 backbone
    """
    arch_zoo = {
        **dict.fromkeys(
            ['t', 'ti', 'tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
                'feedforward_channels': int(192 * 4 * 2 / 3)
=======
                'mlp_ratio': 4 * 2 / 3
>>>>>>> feat: add eva02 backbone
            }),
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
                'feedforward_channels': int(384 * 4 * 2 / 3)
=======
                'mlp_ratio': 4 * 2 / 3
>>>>>>> feat: add eva02 backbone
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
                'feedforward_channels': int(768 * 4 * 2 / 3)
=======
                'mlp_ratio': 4 * 2 / 3
>>>>>>> feat: add eva02 backbone
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
                'feedforward_channels': int(1024 * 4 * 2 / 3)
=======
                'mlp_ratio': 4 * 2 / 3
>>>>>>> feat: add eva02 backbone
            })
    }
    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch='tiny',
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
                 sub_ln=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN'),
                 with_cls_token=True,
                 layer_cfgs=dict(),
                 **kwargs):
        # set essential args for Vision Transformer
        kwargs.update(
            arch=arch,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            with_cls_token=with_cls_token)
        super(EVA02, self).__init__(**kwargs)

        self.num_heads = self.arch_settings['num_heads']

        # Set RoPE
        head_dim = self.embed_dims // self.num_heads
        self.rope = RotaryEmbeddingFast(
            embed_dims=head_dim, patch_resolution=self.patch_resolution)
=======
                 img_size=336,
                 patch_size=14,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 drop_path_rate=0.1,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN'),
                 final_norm=True,
                 out_type='cls_token',
                 with_cls_token=True,
                 frozen_stages=-1,
                 sub_ln=False,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(EVA02, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'mlp_ratio'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.num_heads = self.arch_settings['num_heads']
        self.mlp_ratio = self.arch_settings['mlp_ratio']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            kernel_size=patch_size,
            stride=patch_size)
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set cls token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
            self.num_extra_tokens = 1
        elif out_type != 'cls_token':
            self.cls_token = None
            self.num_extra_tokens = 0
        else:
            raise ValueError(
                'with_cls_token must be True when `out_type="cls_token"`.')

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens,
                        self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # Set RoPE
        half_head_dim = self.embed_dims // self.num_heads // 2
        self.rope = RotaryEmbedding(
            embed_dims=half_head_dim, patch_resolution=self.patch_resolution)
>>>>>>> feat: add eva02 backbone

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
=======
                mlp_ratio=self.mlp_ratio,
>>>>>>> feat: add eva02 backbone
                sub_ln=sub_ln,
                norm_cfg=norm_cfg,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_rate=drop_rate,
                qkv_bias=qkv_bias,
                rope=self.rope,
                with_cls_token=with_cls_token,
                drop_path_rate=dpr[i])
            _layer_cfg.update(layer_cfgs[i])
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
            self.layers.append(EVA02EndcoderLayer(**_layer_cfg))

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

=======
            self.layers.append(Block(**_layer_cfg))

        self.frozen_stages = frozen_stages
        if final_norm:
            self.final_norm = build_norm_layer(norm_cfg, self.embed_dims)
        else:
            self.final_norm = nn.Identity()

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

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # Set out indices
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

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    def init_weights(self):
        super(EVA02, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)
            if self.cls_token is not None:
                trunc_normal_(self.cls_token, std=0.02)

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
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

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
        # freeze cls_token
        self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers) and self.final_norm:
            self.final_norm.eval()
            for param in self.final_norm.parameters():
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]

        x, patch_resolution = self.patch_embed(x)
>>>>>>> feat: add eva02 backbone
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
        x = self.pre_norm(x)

=======
>>>>>>> feat: add eva02 backbone
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, patch_resolution)

            if i in self.out_indices:
                out = self._format_output(x, patch_resolution)

                if i == len(self.layers) - 1:
                    if self.out_type == 'featmap':
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
                        if self.final_norm:
                            out = out.permute(0, 2, 3, 1)
                            out = self.ln1(out)
                            out = out.permute(0, 3, 1, 2)
                    else:
                        if self.final_norm:
                            out = self.ln1(out)
=======
                        out = out.permute(0, 2, 3, 1)
                        out = self.final_norm(out)
                        out = out.permute(0, 3, 1, 2)
                    else:
                        out = self.final_norm(out)
>>>>>>> feat: add eva02 backbone

                outs.append(out)

        return tuple(outs)
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
=======

    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return patch_token.mean(dim=1)
>>>>>>> feat: add eva02 backbone

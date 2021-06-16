import copy

import numpy as np
import torch
import torch.nn as nn
from mmcv import ConfigDict
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.registry import (ATTENTION, POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_dropout,
                                         build_positional_encoding,
                                         build_transformer_layer,
                                         build_transformer_layer_sequence)
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@ATTENTION.register_module()
class T2TModuleAttention(BaseModule):
    """MultiHead self-attention in Tokens-to-Token module.

    Args:
        in_dim (int): Dimension of input tokens.
        embed_dims (int): Embedding dimension
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        qkv_bias (bool): Add bias as qkv Linear module parameter.
            Default: False.
        qk_scale (float, optional): scale of the dot products. Default: None.
        attn_drop (float): A Dropout layer on attn output weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer on out. Default: 0.0.
        init_cfg (dict, optional): Initialization config dict.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim) or (n, batch, embed_dim). Add batch_first
            to synchronize with MultiheadAttention in transformer.py mmcv.
            batch_first should be True in T2TModuleAttention.
    """

    def __init__(self,
                 in_dim,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 init_cfg=None,
                 batch_first=True):
        super(T2TModuleAttention, self).__init__(init_cfg)
        assert batch_first is True, \
            'batch_first should be True when using T2TModuleAttention'
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(in_dim, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, *args, **kwargs):
        assert \
            (query is key or torch.equal(query, key)) and \
            (key is value or torch.equal(key, value)), \
            'In self-attn, query == key == value should be satistied.'
        B, N, C = query.shape

        qkv = self.qkv(query).reshape(B, N, 3, self.num_heads,
                                      self.embed_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        # because the original x has different size with current x,
        # use v to do skip connection
        x = v.squeeze(1) + x

        return x


@ATTENTION.register_module()
class T2TBlockAttention(BaseModule):
    """MultiHead self-attention in T2T-ViT backbone.

    Args:
        embed_dims (int): Embedding dimension
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        qkv_bias (bool): Add bias as qkv Linear module parameter.
            Default: False.
        qk_scale (float, optional): scale of the dot products. Default: None.
        attn_drop (float): A Dropout layer on attn output weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer on out. Default: 0.0.
        dropout_layer (dict): The dropout_layer used when adding the shortcut.
        init_cfg (dict, optional): Initialization config dict.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim) or (n, batch, embed_dim). Add batch_first
            to synchronize with MultiheadAttention in transformer.py mmcv.
            batch_first should be True in T2TBlockAttention.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True):
        super(T2TBlockAttention, self).__init__(init_cfg)
        assert batch_first is True, \
            'batch_first should be True when using T2TBlockAttention'
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.init_cfg = init_cfg

    def forward(self, query, key, value, residual=None, *args, **kwargs):
        assert \
            (query is key or torch.equal(query, key)) and \
            (key is value or torch.equal(key, value)), \
            'In self-attn, query == key == value should be satistied.'

        if residual is None:
            residual = query

        B, N, C = query.shape
        qkv = self.qkv(query).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(
                                          2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return residual + self.dropout_layer(out)


@TRANSFORMER_LAYER.register_module()
class TokenTransformerLayer(BaseTransformerLayer):
    """Tokens-to-token Transformer Layer."""

    def __init__(self,
                 norm_cfg=dict(type='LN'),
                 attn_cfgs=None,
                 *args,
                 **kwargs):
        super(TokenTransformerLayer, self).__init__(
            attn_cfgs=attn_cfgs, *args, **kwargs)

        self.norms = ModuleList()
        num_norms = self.operation_order.count('norm')
        for i in range(num_norms):
            if i == 0:
                self.norms.append(
                    build_norm_layer(norm_cfg, attn_cfgs['in_dim'])[1])
            else:
                self.norms.append(
                    build_norm_layer(norm_cfg, attn_cfgs['embed_dims'])[1])

    def forward(self, *args, **kwargs):
        x = super(TokenTransformerLayer, self).forward(*args, **kwargs)
        return x


class T2T_module(BaseModule):
    """Tokens-to-Token module.

    A layer-wise “Tokens-to-Token module” (T2T_module) to model the local
    structure information of images and reduce the length of tokens
    progressively.
    Args:
        img_size (int): Input image size
        tokens_type (str): Transformer type used in T2T_module,
            transformer or performer.
        in_chans (int): Number of input channels
        embed_dim (int): Embedding dimension
        token_dim (int): Tokens dimension in T2TModuleAttention.
            To overcome the limitations, in T2T module, the channel dimension
            of T2T layer is set small (32 or 64) to reduce MACs
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 img_size=224,
                 tokens_type='transformer',
                 in_chans=3,
                 embed_dim=768,
                 token_dim=64,
                 init_cfg=None):
        super(T2T_module, self).__init__(init_cfg)

        self.embed_dim = embed_dim

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(
                kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            tokentransformer_layer1 = dict(
                type='TokenTransformerLayer',
                attn_cfgs=ConfigDict(
                    type='T2TModuleAttention',
                    in_dim=in_chans * 7 * 7,
                    embed_dims=token_dim,
                    num_heads=1),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=token_dim,
                    feedforward_channels=token_dim,
                    num_fcs=2,
                    act_cfg=dict(type='GELU'),
                    dropout_layer=dict(type='DropPath', drop_prob=0.)),
                operation_order=('norm', 'self_attn', 'norm', 'ffn'),
                batch_first=True)
            self.attention1 = build_transformer_layer(tokentransformer_layer1)
            tokentransformer_layer2 = copy.deepcopy(tokentransformer_layer1)
            tokentransformer_layer2['attn_cfgs']['in_dim'] = token_dim * 3 * 3
            self.attention2 = build_transformer_layer(tokentransformer_layer2)

            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        # there are 3 soft split, stride are 4,2,2 seperately
        self.num_patches = (img_size // (4 * 2 * 2)) * (
            img_size // (4 * 2 * 2))

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(query=x, key=None, value=None)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)),
                                      int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(query=x, key=None, value=None)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)),
                                      int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


@TRANSFORMER_LAYER.register_module()
class T2TTransformerEncoderLayer(BaseTransformerLayer):
    """Implements transformer layer in T2T-ViT backbone."""

    def __init__(self, *args, **kwargs):
        super(T2TTransformerEncoderLayer, self).__init__(*args, **kwargs)
        assert len(self.operation_order) == 4
        assert set(self.operation_order) == set(['self_attn', 'norm', 'ffn'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class T2TTransformerEncoder(TransformerLayerSequence):
    """Transformer layers of T2T-ViT backbone.

    Args:
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
        drop_path_rate (float): Drop path probability of a drop path layer.
            The drop path probabilities in encoder layers are evenly spaced
            from 0 to drop_path_rate, inclusive. Default: 0.0
    """

    def __init__(
            self,
            coder_norm_cfg=dict(type='LN'),
            drop_path_rate=0.,
            *args,
            **kwargs,
    ):
        super(T2TTransformerEncoder, self).__init__(*args, **kwargs)
        if coder_norm_cfg is not None:
            self.coder_norm = build_norm_layer(
                coder_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify coder_norm_cfg'
            self.coder_norm = None

        self.drop_path_rate = drop_path_rate
        self.set_droppath_rate()

    def set_droppath_rate(self):
        dpr = [
            x.item()
            for x in torch.linspace(0, self.drop_path_rate, self.num_layers)
        ]
        for i, layer in enumerate(self.layers):
            for module in layer.modules():
                if isinstance(module, DropPath):
                    module.drop_prob = dpr[i]

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(T2TTransformerEncoder, self).forward(*args, **kwargs)
        if self.coder_norm is not None:
            x = self.coder_norm(x)
        return x


@POSITIONAL_ENCODING.register_module()
class SinusoidEncoding(object):

    def __init__(self):
        super(SinusoidEncoding, self).__init__()

    def __call__(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)


@BACKBONES.register_module()
class T2T_ViT(BaseBackbone):
    """Tokens-to-Token Vision Transformers (T2T-ViT)

    A PyTorch impl of : `Tokens-to-Token ViT: Training Vision Transformers
    from Scratch on ImageNet` - https://arxiv.org/abs/2101.11986

    Args:
        t2t_module (dict): Config of Tokens-to-Token module
        encoder (dict): Config of T2T-ViT backbone
        drop_rate (float): Probability of an element to be zeroed. Default 0.0.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 t2t_module=dict(
                     img_size=224,
                     tokens_type='transformer',
                     in_chans=3,
                     embed_dim=768,
                     token_dim=64),
                 encoder=dict(
                     type='T2TTransformerEncoder',
                     transformerlayers=None,
                     num_layers=12,
                     coder_norm_cfg=None,
                     drop_path_rate=0.),
                 drop_rate=0.,
                 init_cfg=None):
        super(T2T_ViT, self).__init__(init_cfg)

        self.tokens_to_token = T2T_module(**t2t_module)
        num_patches = self.tokens_to_token.num_patches
        embed_dim = self.tokens_to_token.embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        sinusoid_encoding = build_positional_encoding(
            dict(type='SinusoidEncoding'))
        self.pos_embed = nn.Parameter(
            data=sinusoid_encoding(
                n_position=num_patches + 1, d_hid=embed_dim),
            requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.encoder = build_transformer_layer_sequence(encoder)

        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(query=x, key=None, value=None)

        return x[:, 0]

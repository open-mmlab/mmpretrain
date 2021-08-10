# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner.base_module import BaseModule, ModuleList

from ..builder import BACKBONES
from ..utils import PatchEmbed
from .base_backbone import BaseBackbone


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        num_fcs (int): The number of fully-connected layers for FFNs. Default 2
        qkv_bias (bool): enable bias for qkv if True. Default True
        act_cfg (dict): The activation config for FFNs. Defalut GELU
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim) or (n, batch, embed_dim).
            (batch, n, embed_dim) is common case in CV.  Default to False
        init_cfg (dict, optional): Initialization config dict
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=False,
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.batch_first = batch_first

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = self.attn(self.norm1(x), identity=x)
        x = self.ffn(self.norm2(x), identity=x)
        return x


@BACKBONES.register_module()
class VisionTransformer(BaseBackbone):
    """Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: True.
        patch_cfg (dict): TODO
        layer_cfg (dict): TODO
        init_cfg (dict, optional): Initialization config dict
    """
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
                'qkv_bias': False
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
    }

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 output_cls_token=True,
                 patch_cfg=dict(),
                 layer_cfg=dict(),
                 init_cfg=None):
        super(VisionTransformer, self).__init__(init_cfg)

        arch = arch.lower()
        if isinstance(arch, str):
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']

        _patch_cfg = dict(
            img_size=img_size,
            embed_dims=self.embed_dims,
            conv_cfg=dict(
                type='Conv2d', kernel_size=patch_size, stride=patch_size),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        num_patches = self.patch_embed.num_patches

        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate,
                                             self.arch_settings['num_layers'])
        ]  # stochastic depth decay rule

        self.layers = ModuleList()
        for i in range(self.arch_settings['num_layers']):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=self.arch_settings.get('qkv_bias', True),
                norm_cfg=norm_cfg,
                batch_first=True)
            _layer_cfg.update(layer_cfg)
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super(VisionTransformer, self).init_weights()
        # Modified from ClassyVision
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        patch_resolution = self.patch_embed.patches_resolution

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm1(x)

        B, _, C = x.shape
        patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
        cls_token = x[:, 0]

        if self.output_cls_token:
            out = [patch_token, cls_token]
        else:
            out = patch_token

        return (out, )

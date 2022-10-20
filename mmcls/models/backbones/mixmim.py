# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union
import torch
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.registry import MODELS
import random
from torch.nn import functional as F
from torch import nn
from mmcls.models.utils.attention import WindowMSA
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer
from mmengine.model import BaseModule
from torch.utils.checkpoint import checkpoint

from mmcv.cnn.bricks.transformer import FFN, PatchEmbed, PatchMerging
from mmcv.cnn.bricks.drop import DropPath
from mmcv.cnn import build_norm_layer

from ..utils import build_2d_sincos_position_embedding
from mmcls.models.utils.helpers import to_2tuple





@MODELS.register_module()
class MixMIMTransformer(BaseBackbone):
    arch_settings = {
        **dict.fromkeys(['B', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32]}),
        **dict.fromkeys(['L', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48]}),
        **dict.fromkeys(['H', 'huge'],
                        {'embed_dims': 352,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [11, 22, 44, 88]}),
    }

    def __init__(self,
                 arch='base',
                 mlp_ratio=4,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 window_size=[14, 14, 14, 7],
                 qkv_bias=True,
                 patch_cfg=dict(),
                 norm_cfg=dict(type='LN'),
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 attn_drop_rate=0.0,
                 range_mask_ratio=0.0,
                 use_checkpoint=False,
                 init_cfg: Optional[dict] = None,
                 ) -> None:
        super(MixMIMTransformerFinetune, self).__init__(init_cfg=init_cfg)

        self.embed_dims = self.arch_settings[arch]['embed_dims']
        self.depths = self.arch_settings[arch]['depths']
        self.num_heads = self.arch_settings[arch]['num_heads']

        self.encoder_stride = 32

        self.num_layers = len(self.depths)
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.use_checkpoint = use_checkpoint
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=dict(type='LN'),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(MixMIMLayer(
                embed_dims=int(self.embed_dims * 2 ** i_layer),
                input_resolution=(self.patch_resolution[0] // (2 ** i_layer), self.patch_resolution[1] // (2 ** i_layer)),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                proj_drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=self.dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_cfg=norm_cfg,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=self.use_checkpoint)
            )

        self.num_features = int(self.embed_dims * 2 ** (self.num_layers - 1))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        self.range_mask_ratio = range_mask_ratio
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dims), requires_grad=False)

        _, self.norm = build_norm_layer(
            norm_cfg, self.num_features)

    def init_weights(self):
        super(MixMIMTransformerFinetune, self).init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x: torch.Tensor):

        x, patch_resolution = self.patch_embed(x)

        B, L, _ = x.shape

        x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x, attn_mask=None)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        return [x]
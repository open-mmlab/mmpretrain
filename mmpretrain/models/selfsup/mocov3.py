# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import reduce
from operator import mul
from typing import List, Optional, Union

import torch.nn as nn

from mmcv.cnn.bricks.transformer import PatchEmbed
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import VisionTransformer
from mmpretrain.models.utils import (build_2d_sincos_position_embedding,
                                     to_2tuple)
from mmpretrain.registry import MODELS


@MODELS.register_module()
class MoCoV3ViT(VisionTransformer):
    """Vision Transformer for MoCoV3 pre-training.

    A pytorch implement of: `An Images is Worth 16x16 Words: Transformers for
    Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Part of the code is modified from:
    `<https://github.com/facebookresearch/moco-v3/blob/main/vits.py>`_.

    Args:
        stop_grad_conv1 (bool): whether to stop the gradient of
            convolution layer in `PatchEmbed`. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 stop_grad_conv1: bool = False,
                 frozen_stages: int = -1,
                 norm_eval: bool = False,
                 init_cfg: Optional[Union[dict, List[dict]]] = None,
                 **kwargs) -> None:

        # add MoCoV3 ViT-small arch
        self.arch_zoo.update(
            dict.fromkeys(
                ['mocov3-s', 'mocov3-small'], {
                    'embed_dims': 384,
                    'num_layers': 12,
                    'num_heads': 12,
                    'feedforward_channels': 1536,
                }))

        super().__init__(init_cfg=init_cfg, **kwargs)
        self.patch_size = kwargs['patch_size']
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.init_cfg = init_cfg

        if isinstance(self.patch_embed, PatchEmbed):
            if stop_grad_conv1:
                self.patch_embed.projection.weight.requires_grad = False
                self.patch_embed.projection.bias.requires_grad = False

        self._freeze_stages()

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding, qkv layers and cls
        token."""
        super().init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):

            # Use fixed 2D sin-cos position embedding
            pos_emb = build_2d_sincos_position_embedding(
                patches_resolution=self.patch_resolution,
                embed_dims=self.embed_dims,
                cls_token=True)
            self.pos_embed.data.copy_(pos_emb)
            self.pos_embed.requires_grad = False

            # xavier_uniform initialization for PatchEmbed
            if isinstance(self.patch_embed, PatchEmbed):
                val = math.sqrt(
                    6. / float(3 * reduce(mul, to_2tuple(self.patch_size), 1) +
                               self.embed_dims))
                nn.init.uniform_(self.patch_embed.projection.weight, -val, val)
                nn.init.zeros_(self.patch_embed.projection.bias)

            # initialization for linear layers
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if 'qkv' in name:
                        # treat the weights of Q, K, V separately
                        val = math.sqrt(
                            6. /
                            float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                        nn.init.uniform_(m.weight, -val, val)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            nn.init.normal_(self.cls_token, std=1e-6)

    def _freeze_stages(self) -> None:
        """Freeze patch_embed layer, some parameters and stages."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i == (self.num_layers) and self.final_norm:
                for param in getattr(self, 'norm1').parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

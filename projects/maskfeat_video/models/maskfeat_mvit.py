# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.models import MViT
from mmaction.models.backbones.mvit import resize_pos_embed

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MaskFeatMViT(MViT):

    arch_zoo = {
        'maskfeat-small': {
            'embed_dims': 96,
            'num_layers': 16,
            'num_heads': 1,
            'downscale_indices': [1, 3],
            'dim_mul_indices': [1, 3, 14]
        },
        'maskfeat-large': {
            'embed_dims': 144,
            'num_layers': 48,
            'num_heads': 2,
            'downscale_indices': [2, 8],
            'dim_mul_indices': [2, 8, 44]
        },
    }

    def __init__(
        self,
        arch: str = 'base',
        spatial_size: int = 224,
        temporal_size: int = 16,
        in_channels: int = 3,
        out_scales: Union[int, Sequence[int]] = -1,
        drop_path_rate: float = 0,
        use_abs_pos_embed: bool = False,
        interpolate_mode: str = 'trilinear',
        pool_kernel: tuple = (3, 3, 3),
        dim_mul: int = 2,
        head_mul: int = 2,
        adaptive_kv_stride: tuple = (1, 8, 8),
        rel_pos_embed: bool = True,
        residual_pooling: bool = True,
        dim_mul_in_attention: bool = True,
        with_cls_token: bool = True,
        output_cls_token: bool = True,
        rel_pos_zero_init: bool = False,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        norm_cfg: dict = dict(type='LN', eps=1e-6),
        patch_cfg: dict = dict(
            kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3)),
        init_cfg: Optional[Union[dict, List[dict]]] = [
            dict(type='TruncNormal', layer=['Conv2d', 'Conv3d'], std=0.02),
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.02),
        ]
    ) -> None:
        super().__init__(
            arch=arch,
            spatial_size=spatial_size,
            temporal_size=temporal_size,
            in_channels=in_channels,
            out_scales=out_scales,
            drop_path_rate=drop_path_rate,
            use_abs_pos_embed=use_abs_pos_embed,
            interpolate_mode=interpolate_mode,
            pool_kernel=pool_kernel,
            dim_mul=dim_mul,
            head_mul=head_mul,
            adaptive_kv_stride=adaptive_kv_stride,
            rel_pos_embed=rel_pos_embed,
            residual_pooling=residual_pooling,
            dim_mul_in_attention=dim_mul_in_attention,
            with_cls_token=with_cls_token,
            output_cls_token=output_cls_token,
            rel_pos_zero_init=rel_pos_zero_init,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            patch_cfg=patch_cfg,
            init_cfg=init_cfg)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        self.patch_stride = patch_cfg['stride']

    def init_weights(self) -> None:
        """Initialize mask token and cls token."""
        super().init_weights()
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor]:

        x, patch_resolution = self.patch_embed(x)
        B, L, C = x.shape
        T, H, W = patch_resolution

        mask_tokens = self.mask_token.expand(B, L, -1)
        mask = F.interpolate(mask.float(), size=(H, W))
        mask = mask.flatten(1).unsqueeze(-1)
        x = x * (1 - mask) + mask_tokens * mask

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens)

        # if not self.with_cls_token:
        #     # Remove class token for transformer encoder input
        #     x = x[:, 1:]

        outs = []
        self.out_patch_resolution = []
        for i, block in enumerate(self.blocks):
            x, patch_resolution = block(x, patch_resolution)

            if i in self.stage_indices:
                stage_index = self.stage_indices[i]
                if stage_index in self.out_scales:
                    self.out_patch_resolution.append(patch_resolution)
                    x = getattr(self, f'norm{stage_index}')(x)
                    if not self.output_cls_token:
                        out = x[:, 1:]
                    else:
                        out = x
                    outs.append(out)

        return tuple(outs)

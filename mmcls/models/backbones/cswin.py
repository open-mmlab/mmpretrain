# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops.layers.torch import Rearrange
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN
from mmengine.utils import to_2tuple

from mmcls.registry import MODELS
from ..utils import LePEAttention
from .base_backbone import BaseBackbone


class CSWinBlock(nn.Module):

    def __init__(self,
                 dim,
                 reso,
                 num_heads,
                 split_size=7,
                 mlp_ratio=4.,
                 num_fcs=2,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim,
                    resolution=self.patches_resolution,
                    idx=-1,
                    split_size=split_size,
                    num_heads=num_heads,
                    dim_out=dim,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop) for i in range(self.branch_num)
            ])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2,
                    resolution=self.patches_resolution,
                    idx=i,
                    split_size=split_size,
                    num_heads=num_heads // 2,
                    dim_out=dim // 2,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop) for i in range(self.branch_num)
            ])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = FFN(
            embed_dims=dim,
            feedforward_channels=mlp_hidden_dim,
            num_fcs=num_fcs,
            ffn_drop=drop,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path),
            act_cfg=act_cfg)

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, 'flatten img_tokens has wrong size'
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = self.ffn(self.norm2(x), identity=x)

        return x, self.patches_resolution


class Merge_Block(nn.Module):

    def __init__(self, dim, dim_out, norm_cfg=dict(type='LN')):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = build_norm_layer(norm_cfg, dim_out)[1]

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x


@MODELS.register_module()
class CSWinTransformer(BaseBackbone):
    """Vision Transformer with support for patch or hybrid CNN input stage."""
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 64,
                         'depths':     [1, 2, 21, 1],
                         'num_heads':  [2, 4, 8, 16]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 64,
                         'depths':     [2, 4, 32,  2],
                         'num_heads':  [2, 4, 8, 16]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 96,
                         'depths':     [2, 4, 32,  2],
                         'num_heads':  [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 144,
                         'depths':     [2,  4, 32,  2],
                         'num_heads':  [6, 12, 24, 24]}),
    }  # yapf: disable

    def __init__(self,
                 arch='tiny',
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 split_size=[3, 5, 7],
                 out_indices=(3, ),
                 frozen_stages=-1,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 use_chk=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):

        super(CSWinTransformer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        embed_dims = self.num_features = self.embed_dims = self.arch_settings[
            'embed_dims']
        depths = self.depths = self.arch_settings['depths']
        heads = self.arch_settings['num_heads']

        self.frozen_stages = frozen_stages

        self.out_indices = out_indices

        self.img_size = to_2tuple(img_size)

        self.use_chk = use_chk
        self.num_classes = num_classes

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims, 7, 4, 2),
            Rearrange(
                'b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dims))

        curr_dim = embed_dims
        self.num_features = [curr_dim]
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depths))
        ]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim,
                num_heads=heads[0],
                reso=img_size // 4,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                split_size=split_size[0],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_cfg=dict(type='LN')) for i in range(depths[0])
        ])

        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.num_features.append(curr_dim)
        self.stage2 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim,
                num_heads=heads[1],
                reso=img_size // 8,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                split_size=split_size[1],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depths[:1]) + i],
                norm_cfg=dict(type='LN')) for i in range(depths[1])
        ])

        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.num_features.append(curr_dim)
        temp_stage3 = []
        temp_stage3.extend([
            CSWinBlock(
                dim=curr_dim,
                num_heads=heads[2],
                reso=img_size // 16,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                split_size=split_size[2],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depths[:2]) + i],
                norm_cfg=dict(type='LN')) for i in range(depths[2])
        ])

        self.stage3 = nn.ModuleList(temp_stage3)

        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.num_features.append(curr_dim)
        self.stage4 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim,
                num_heads=heads[3],
                reso=img_size // 32,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                split_size=split_size[-1],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depths[:-1]) + i],
                norm_cfg=dict(type='LN'),
                last_stage=True) for i in range(depths[-1])
        ])

        # self.norm = norm_layer(curr_dim)
        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg,
                                              self.num_features[i])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm{i}', norm_layer)

        if self.frozen_stages > 0:
            self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stage1_conv_embed.eval()
            for param in self.stage1_conv_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            for param in getattr(self, f'merge{i}').parameters():
                param.requires_grad = False

        for i in self.out_indices:
            if i <= self.frozen_stages:
                for param in getattr(self, f'norm{i}').parameters():
                    param.requires_grad = False

    def init_weights(self):
        super(CSWinTransformer, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

    def forward_features(self, x):
        x = self.stage1_conv_embed(x)

        outs = []

        for blk in self.stage1:
            if self.use_chk:
                x, hw_shape = checkpoint.checkpoint(blk, x)
            else:
                x, hw_shape = blk(x)

        if 0 in self.out_indices:
            norm_layer = getattr(self, f'norm{0}')
            out = norm_layer(x)
            out = out.view(-1, *(hw_shape, hw_shape),
                           self.num_features[0]).permute(0, 3, 1,
                                                         2).contiguous()
            outs.append(out)

        for i, (pre, blocks) in enumerate(
                zip([self.merge1, self.merge2, self.merge3],
                    [self.stage2, self.stage3, self.stage4])):
            x = pre(x)
            for blk in blocks:
                if self.use_chk:
                    x, hw_shape = checkpoint.checkpoint(blk, x)
                else:
                    x, hw_shape = blk(x)
            if i + 1 in self.out_indices:
                norm_layer = getattr(self, f'norm{i+1}')
                out = norm_layer(x)
                out = out.view(-1, *(hw_shape, hw_shape),
                               self.num_features[i + 1]).permute(
                                   0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def forward(self, x):
        outs = self.forward_features(x)
        return outs


def _conv_filter(state_dict, patch_size=16):
    """convert patch embedding weight from manual patchify + linear proj to
    conv."""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

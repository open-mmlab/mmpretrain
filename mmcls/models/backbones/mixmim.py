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




<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> add mixmim backbone
class MixMIMWindowAttention(WindowMSA):
    def __init__(self,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__(
                 embed_dims=embed_dims,
                 window_size=window_size,
                 num_heads=num_heads,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 attn_drop=attn_drop_rate,
                 proj_drop=proj_drop_rate,
                 init_cfg=init_cfg)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.reshape(B_, 1, 1, N)
            mask_new = mask * mask.transpose(2, 3) + (1 - mask) * (1 - mask).transpose(2, 3)
            mask_new = 1 - mask_new

            if mask_new.dtype == torch.float16:
                attn = attn - 65500 * mask_new
            else:
                attn = attn - 1e30 * mask_new

            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixMIMBlock(TransformerEncoderLayer):

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 num_fcs=2,
                 qkv_bias=True,
                 proj_drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:

        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=int(mlp_ratio * embed_dims),
            drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_fcs=num_fcs,
            qkv_bias=qkv_bias,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(self.input_resolution)

        self.attn = MixMIMWindowAttention(embed_dims=embed_dims, window_size=to_2tuple(self.window_size),
                                          num_heads=num_heads, qkv_bias=qkv_bias,
                                          attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    @staticmethod
    def window_reverse(windows, H, W, window_size):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    def forward(self, x, attn_mask=None):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = self.window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(B, 1, 1)   # B, N, 1
            attn_mask = attn_mask.view(B, H, W, 1)
            attn_mask = self.window_partition(attn_mask, self.window_size)
            attn_mask = attn_mask.view(-1, self.window_size * self.window_size, 1)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, H, W, self.window_size)  # B H' W' C

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        x = self.ffn(self.norm2(x))  # ffn contains DropPath

        return x

class MixMIMLayer(BaseModule):

    def __init__(self, embed_dims, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, proj_drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=[0.], norm_cfg=dict(type='LN'), downsample=None,
                 use_checkpoint=False, init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                MixMIMBlock(
                    embed_dims=embed_dims, input_resolution=input_resolution, num_heads=num_heads,
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_drop_rate=proj_drop_rate, attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate[i],
                    norm_cfg=norm_cfg)
            )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(in_channels=embed_dims, out_channels=2 * embed_dims, norm_cfg=norm_cfg)
        else:
            self.downsample = None

    def forward(self, x, attn_mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask=attn_mask)
        if self.downsample is not None:
            x, _ = self.downsample(x, self.input_resolution)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"




<<<<<<< HEAD
class MixMIMWindowAttention(WindowMSA):
    def __init__(self,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__(
                 embed_dims=embed_dims,
                 window_size=window_size,
                 num_heads=num_heads,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 attn_drop=attn_drop_rate,
                 proj_drop=proj_drop_rate,
                 init_cfg=init_cfg)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.reshape(B_, 1, 1, N)
            mask_new = mask * mask.transpose(2, 3) + (1 - mask) * (1 - mask).transpose(2, 3)
            mask_new = 1 - mask_new

            if mask_new.dtype == torch.float16:
                attn = attn - 65500 * mask_new
            else:
                attn = attn - 1e30 * mask_new

            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixMIMBlock(TransformerEncoderLayer):

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 num_fcs=2,
                 qkv_bias=True,
                 proj_drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:

        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=int(mlp_ratio * embed_dims),
            drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_fcs=num_fcs,
            qkv_bias=qkv_bias,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(self.input_resolution)

        self.attn = MixMIMWindowAttention(embed_dims=embed_dims, window_size=to_2tuple(self.window_size),
                                          num_heads=num_heads, qkv_bias=qkv_bias,
                                          attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    @staticmethod
    def window_reverse(windows, H, W, window_size):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    def forward(self, x, attn_mask=None):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = self.window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(B, 1, 1)   # B, N, 1
            attn_mask = attn_mask.view(B, H, W, 1)
            attn_mask = self.window_partition(attn_mask, self.window_size)
            attn_mask = attn_mask.view(-1, self.window_size * self.window_size, 1)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, H, W, self.window_size)  # B H' W' C

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        x = self.ffn(self.norm2(x))  # ffn contains DropPath

        return x

class MixMIMLayer(BaseModule):

    def __init__(self, embed_dims, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, proj_drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=[0.], norm_cfg=dict(type='LN'), downsample=None,
                 use_checkpoint=False, init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                MixMIMBlock(
                    embed_dims=embed_dims, input_resolution=input_resolution, num_heads=num_heads,
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_drop_rate=proj_drop_rate, attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate[i],
                    norm_cfg=norm_cfg)
            )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(in_channels=embed_dims, out_channels=2 * embed_dims, norm_cfg=norm_cfg)
        else:
            self.downsample = None

    def forward(self, x, attn_mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask=attn_mask)
        if self.downsample is not None:
            x, _ = self.downsample(x, self.input_resolution)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"




=======
>>>>>>> add mixmim backbone
=======
>>>>>>> add mixmim backbone

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
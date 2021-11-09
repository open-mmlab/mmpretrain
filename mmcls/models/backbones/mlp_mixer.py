import collections

import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.runner.base_module import BaseModule, Sequential
from mmcv.utils.misc import to_2tuple

from mmcls.models.utils import PatchEmbed
from ..builder import BACKBONES


class MixerBlock(BaseModule):
    """Mlp-Mixer basic block.

    Basic module of `MLP-Mixer: An all-MLP Architecture for Vision
    <https://arxiv.org/pdf/2105.01601.pdf>`_

    Some code is borrowed and modified from timm.

    Args:
        embed_dims (int): Embedding dimension.
        num_patches (int): Number of patches.
        mlp_ratio (tuple | int): Multiplier based on embed_dims for
            hidden widths of token-mixing and channel-mixing MLPs. Defaults
            to (0.5, 4.0)
        norm_cfg (dict): Configuration for the normalization layer. Defaults to
            dict(type='LN', eps=1e-6).
        act_layer_cfg (dict): Configuration for the activation layer. Defaults
            to dict(type='GELU').
        drop_rate (float): Dropout rate for the dropout layer in MixerBlock.
            Defaults to 0.
        drop_path_rate (float): Drop path rate for the DropPath (Stochastic
            Depth) in Mixer. Defaults to 0.
    """

    def __init__(self,
                 embed_dims,
                 num_patches,
                 mlp_ratio=(0.5, 4.0),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_layer_cfg=dict(type='GELU'),
                 drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        tokens_dim, channels_dim = [
            int(x * embed_dims) for x in to_2tuple(mlp_ratio)
        ]
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.mlp_tokens = self.build_mlp(
            num_patches,
            tokens_dim,
            act_layer_cfg=act_layer_cfg,
            drop_rate=drop_rate)
        self.drop_path = DropPath(drop_prob=drop_path_rate) \
            if drop_path_rate > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.mlp_channels = self.build_mlp(
            embed_dims,
            channels_dim,
            act_layer_cfg=act_layer_cfg,
            drop_rate=drop_rate)

    @staticmethod
    def build_mlp(in_features,
                  hidden_features=None,
                  out_features=None,
                  act_layer_cfg=None,
                  drop_rate=0.):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        mlp = Sequential(*[
            nn.Linear(in_features, hidden_features),
            build_activation_layer(act_layer_cfg),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop_rate)
        ])

        return mlp

    def forward(self, x):
        x = x + self.drop_path(
            self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


@BACKBONES.register_module()
class MlpMixer(BaseModule):
    """Mlp-Mixer backbone.

    Pytorch implementation of `MLP-Mixer: An all-MLP Architecture for Vision
    <https://arxiv.org/pdf/2105.01601.pdf>`_

    Some code is borrowed and modified from timm.

    Args:
        img_size (int): Input image size. Defaults to 224.
        in_channels (int): Number of input channels. Defaults to 3.
        patch_size (int): Size of patch. Defaults to 16.
        num_blocks (int): Num of mixer block. Defaults to 8.
        embed_dims (int): Embedding dimension.
        mlp_ratio (tuple | int): Multiplier for the D_s and D_c. Defaults
            to (0.5, 4.0)
        norm_cfg (dict): Configuration for the normalization layer. Defaults to
            dict(type='LN', eps=1e-6).
        act_layer_cfg (dict): Configuration for the activation layer. Defaults
            to dict(type='GELU').
        drop_rate (float): Dropout rate for the dropout layer in MixerBlock.
            Defaults to 0.
        drop_path_rate (float | Sequence[float]): Drop path rate for the
            DropPath (Stochastic Depth) in Mixer. Defaults to 0.
        stem_norm (bool): If use normalization in the stem part. Defaults to
            False.
        out_indices (Sequence[int]): Output from which stages. Defaults to
            (-1, )
    """

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 patch_size=16,
                 num_blocks=8,
                 embed_dims=512,
                 mlp_ratio=(0.5, 4.0),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_layer_cfg=dict(type='GELU'),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 stem_norm=False,
                 out_indices=(-1, )):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_blocks = num_blocks

        for i, idx in enumerate(out_indices):
            converted_idx = idx if idx >= 0 else num_blocks + idx
            assert 0 <= converted_idx < num_blocks, \
                f'Invalid out_indices {idx} at position {i}'

        self.out_indices = [
            idx if idx >= 0 else num_blocks + idx for idx in out_indices
        ]

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            conv_cfg=dict(kernel_size=patch_size, stride=patch_size),
            in_channels=in_channels,
            embed_dims=embed_dims,
            norm_cfg=norm_cfg if stem_norm else None)

        assert isinstance(drop_path_rate, (float, collections.abc.Iterable)), (
            'drop_path_rate should be one of [float, list, tuple], ',
            f'but get {type(drop_path_rate)}')

        if isinstance(drop_path_rate, float):
            drop_path_rate = [drop_path_rate for _ in range(num_blocks)]

        assert len(drop_path_rate) == num_blocks, (
            'The drop_path_rate should be a single value or ',
            f'have {num_blocks} elements, but got {len(drop_path_rate)}')

        self.blocks = Sequential(*[
            MixerBlock(
                embed_dims,
                self.patch_embed.num_patches,
                mlp_ratio,
                norm_cfg=norm_cfg,
                act_layer_cfg=act_layer_cfg,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate[idx])
            for idx in range(num_blocks)
        ])
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x):
        x = self.patch_embed(x)
        out = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.num_blocks - 1:
                x = self.norm(x)
                x = x.mean(dim=1)
            if i in self.out_indices:
                out.append(x)
        return tuple(out)

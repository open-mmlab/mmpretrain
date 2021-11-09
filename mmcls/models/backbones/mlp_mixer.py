import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner.base_module import BaseModule, Sequential
from mmcv.utils.misc import to_2tuple

from mmcls.models.utils import PatchEmbed
from ..builder import BACKBONES


class MixerBlock(nn.Module):
    """MlpMixer block.

    Basic module of `MLP-Mixer: An all-MLP Architecture for Vision
    <https://arxiv.org/pdf/2105.01601.pdf>`_

    Some code is borrowed and modified from timm.

    Args:
        dim (int): Embedding dimension.
        seq_len (int): Length of the sequence, mally equivalent to um patches.
        mlp_ratio (tuple | int): Multiplier for the D_s and D_c. Default
            to (0.5, 4.0)
        norm_cfg (dict): Configuration for the normalization layer. Default to
            dict(type='LN', eps=1e-6).
        act_layer_cfg (dict): Configuration for the activation layer. Default
            to dict(type='GELU').
        drop_rate (float): Dropout rate for the dropout layer in MixerBlock.
            Default to 0.
        drop_path_rate (float | list): Drop path rate for the DropPath (
            Stochastic Depth) in Mixer. Default to 0.
    """

    def __init__(self,
                 dim,
                 seq_len,
                 mlp_ratio=(0.5, 4.0),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_layer_cfg=dict(type='GELU'),
                 drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp_tokens = self.__build_mlp(
            seq_len,
            tokens_dim,
            act_layer_cfg=act_layer_cfg,
            drop_rate=drop_rate)
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp_channels = self.__build_mlp(
            dim,
            channels_dim,
            act_layer_cfg=act_layer_cfg,
            drop_rate=drop_rate)

    def __build_mlp(self,
                    in_features,
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
    """MlpMixer backbone.

    Pytorch implementation of `MLP-Mixer: An all-MLP Architecture for Vision
    <https://arxiv.org/pdf/2105.01601.pdf>`_

    Some code is borrowed and modified from timm.

    Args:
        img_size (int): Input image size. Default to 224.
        in_channels (int): Number of input channels. Default to 3.
        patch_size (int): Size of patch. Default to 16.
        num_blocks (int): Num of mixer block. Default to 8.
        embed_dims (int): Embedding dimension.
        mlp_ratio (tuple | int): Multiplier for the D_s and D_c. Default
            to (0.5, 4.0)
        norm_cfg (dict): Configuration for the normalization layer. Default to
            dict(type='LN', eps=1e-6).
        act_layer_cfg (dict): Configuration for the activation layer. Default
            to dict(type='GELU').
        drop_rate (float): Dropout rate for the dropout layer in MixerBlock.
            Default to 0.
        drop_path_rate (float | list): Drop path rate for the DropPath (
            Stochastic Depth) in Mixer. Default to 0.
        stem_norm (bool): If use normalization in the stem part. Default to
            False.
    """

    def __init__(
        self,
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
    ):
        super().__init__()

        self.num_features = self.embed_dims = embed_dims

        self.stem = PatchEmbed(
            img_size=img_size,
            conv_cfg=dict(kernel_size=patch_size, stride=patch_size),
            in_channels=in_channels,
            embed_dims=embed_dims,
            norm_cfg=norm_cfg if stem_norm else None)

        if isinstance(drop_path_rate, (int, float)):
            drop_path_rate = [drop_path_rate for _ in range(num_blocks)]

        assert len(
            drop_path_rate
        ) == num_blocks, f'The drop_path_rate should be a single value or ' \
                         f'have {num_blocks} elements, but got ' \
                         f'{len(drop_path_rate)}'

        self.blocks = Sequential(*[
            MixerBlock(
                embed_dims,
                self.stem.num_patches,
                mlp_ratio,
                norm_cfg=norm_cfg,
                act_layer_cfg=act_layer_cfg,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate[idx])
            for idx in range(num_blocks)
        ])
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return (x, )


def convert_weights(weight):
    """Weight Converter.

    Converts the weights from timm to mmcls

    Args:
        weight (dict): weight dict from timm

    Returns: converted weight dict for mmcls
    """
    result = dict()
    result['meta'] = dict()
    temp = dict()
    mapping = {
        'proj': 'projection',
        'mlp_tokens.fc1': 'mlp_tokens.0',
        'mlp_tokens.fc2': 'mlp_tokens.3',
        'mlp_channels.fc1': 'mlp_channels.0',
        'mlp_channels.fc2': 'mlp_channels.3',
    }
    for k, v in weight.items():
        for mk, mv in mapping.items():
            if mk in k:
                k = k.replace(mk, mv)
        if k.startswith('head.'):
            temp['head.fc.' + k[5:]] = v
        else:
            temp['backbone.' + k] = v
    result['state_dict'] = temp
    return result

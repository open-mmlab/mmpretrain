import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner.base_module import BaseModule, Sequential
from mmcv.utils.misc import to_2tuple

from mmcls.models.utils import PatchEmbed
from ..builder import BACKBONES


class MixerBlock(nn.Module):
    """Residual Block w/ token mixing and channel MLPs.

    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision'
     - https://arxiv.org/abs/2105.01601
    """

    def __init__(self,
                 dim,
                 seq_len,
                 mlp_ratio=(0.5, 4.0),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_layer_cfg=dict(type='GELU'),
                 drop=0.,
                 drop_path=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp_tokens = self.__build_mlp(
            seq_len, tokens_dim, act_layer_cfg=act_layer_cfg, drop=drop)
        self.drop_path = build_dropout(
            dict(type='DropPath',
                 drop_prob=drop_path)) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp_channels = self.__build_mlp(
            dim, channels_dim, act_layer_cfg=act_layer_cfg, drop=drop)

    def __build_mlp(self,
                    in_features,
                    hidden_features=None,
                    out_features=None,
                    act_layer_cfg=None,
                    drop=0.):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        mlp = Sequential(*[
            nn.Linear(in_features, hidden_features),
            build_activation_layer(act_layer_cfg),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        ])

        return mlp

    def forward(self, x):
        x = x + self.drop_path(
            self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


@BACKBONES.register_module()
class MlpMixer(BaseModule):

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
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = Sequential(*[
            MixerBlock(
                embed_dims,
                self.stem.num_patches,
                mlp_ratio,
                norm_cfg=norm_cfg,
                act_layer_cfg=act_layer_cfg,
                drop=drop_rate,
                drop_path=drop_path_rate) for _ in range(num_blocks)
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
        weight: weight dict from timm

    Returns: transformed mmcls weight dict
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

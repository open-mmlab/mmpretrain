import math
from functools import partial

import torch.nn as nn
from timm.models.helpers import named_apply
from timm.models.layers import Mlp, PatchEmbed
from timm.models.mlp_mixer import MixerBlock, _init_weights

from ..builder import BACKBONES


@BACKBONES.register_module()
class MlpMixer(nn.Module):

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        patch_size=16,
        num_blocks=8,
        embed_dim=512,
        mlp_ratio=(0.5, 4.0),
        block_layer=MixerBlock,
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop_rate=0.,
        drop_path_rate=0.,
        nlhb=False,
        stem_norm=False,
    ):
        super().__init__()
        # num_features for consistency with other timm models
        self.num_features = self.embed_dim = embed_dim

        self.stem = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if stem_norm else None)
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim,
                self.stem.num_patches,
                mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=drop_rate,
                drop_path=drop_path_rate) for _ in range(num_blocks)
        ])
        self.norm = norm_layer(embed_dim)

        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(
            partial(_init_weights, head_bias=head_bias),
            module=self)  # depth-first

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        result = []
        x = self.forward_features(x)
        result.append(x)
        return tuple(result)

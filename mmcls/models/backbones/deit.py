# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import BACKBONES
from .vision_transformer import VisionTransformer


@BACKBONES.register_module()
class DistilledVisionTransformer(VisionTransformer):

    def __init__(self, *args, **kwargs):
        super(DistilledVisionTransformer, self).__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 2, self.embed_dims))

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        patch_resolution = self.patch_embed.patches_resolution

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x[:, 2:].reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)
                cls_token = x[:, 0]
                dist_token = x[:, 1]
                if self.output_cls_token:
                    out = [patch_token, cls_token, dist_token]
                else:
                    out = patch_token
                outs.append(out)

        return tuple(outs)

    def init_weights(self):
        super(DistilledVisionTransformer, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.dist_token, std=0.02)

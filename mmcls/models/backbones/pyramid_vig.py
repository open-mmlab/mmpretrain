# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import Sequential

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.registry import MODELS
from .vig import FFN, Grapher


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = Sequential(
            build_conv_layer(
                None, in_dim, out_dim // 2, 3, stride=2, padding=1),
            build_norm_layer(dict(type='BN'), out_dim // 2)[1],
            build_activation_layer(act),
            build_conv_layer(
                None, out_dim // 2, out_dim, 3, stride=2, padding=1),
            build_norm_layer(dict(type='BN'), out_dim)[1],
            build_activation_layer(act),
            build_conv_layer(None, out_dim, out_dim, 3, stride=1, padding=1),
            build_norm_layer(dict(type='BN'), out_dim)[1],
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """Convolution-based downsample."""

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = Sequential(
            build_conv_layer(None, in_dim, out_dim, 3, stride=2, padding=1),
            build_norm_layer(dict(type='BN'), out_dim)[1],
        )

    def forward(self, x):
        x = self.conv(x)
        return x


@MODELS.register_module()
class PyramidVig(BaseBackbone):
    # blocks, channels
    arch_settings = {
        'tiny': [[2, 2, 6, 2], [48, 96, 240, 384]],
        'small': [[2, 2, 6, 2], [80, 160, 400, 640]],
        'medium': [[2, 2, 16, 2], [96, 192, 384, 768]],
        'base': [[2, 2, 18, 2], [128, 256, 512, 1024]]
    }

    def __init__(self,
                 arch,
                 k=9,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN'),
                 graph_conv_bias=True,
                 graph_conv_type='mr',
                 epsilon=0.2,
                 use_stochastic=False,
                 drop_path=0.,
                 norm_eval=False,
                 frozen_stages=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        arch = self.arch_settings[arch]
        blocks = arch[0]
        self.n_blocks = sum(blocks)
        channels = arch[1]
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)
               ]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)
                   ]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(out_dim=channels[0], act=act_cfg)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, channels[0], 224 // 4, 224 // 4))
        HW = 224 // 4 * 224 // 4

        self.stage_blocks = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.stage_blocks.append(
                    Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.stage_blocks += [
                    Sequential(
                        Grapher(
                            channels[i],
                            num_knn[idx],
                            min(idx // 4 + 1, max_dilation),
                            graph_conv_type,
                            act_cfg,
                            norm_cfg,
                            graph_conv_bias,
                            use_stochastic,
                            epsilon,
                            reduce_ratios[i],
                            n=HW,
                            drop_path=dpr[idx],
                            relative_pos=True),
                        FFN(channels[i],
                            channels[i] * 4,
                            act=act_cfg,
                            drop_path=dpr[idx]))
                ]
                idx += 1
        self.stage_blocks = Sequential(*self.stage_blocks)
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

    def forward(self, inputs):
        outs = []
        x = self.stem(inputs) + self.pos_embed

        for i in range(len(self.stage_blocks)):
            x = self.stage_blocks[i](x)
            outs.append(x)

        x = F.adaptive_avg_pool2d(x, 1)
        outs.append(x)
        return outs

    def _freeze_stages(self):
        self.stem.eval()
        for i in range(self.frozen_stages):
            m = self.stage_blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(PyramidVig, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

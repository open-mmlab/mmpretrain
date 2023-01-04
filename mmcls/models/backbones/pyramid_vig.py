# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from mmcls.registry import MODELS
from .vig import FFN, Grapher, act_layer


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """Convolution-based downsample."""

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


@MODELS.register_module()
class Pyramid_Vig(torch.nn.Module):

    def __init__(self, channels, k, act, norm, bias, epsilon, use_stochastic,
                 conv, drop_path, dropout, blocks, n_classes):
        super(Pyramid_Vig, self).__init__()

        self.n_blocks = sum(blocks)
        channels = channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)
               ]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)
                   ]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, channels[0], 224 // 4, 224 // 4))
        HW = 224 // 4 * 224 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(
                        Grapher(
                            channels[i],
                            num_knn[idx],
                            min(idx // 4 + 1, max_dilation),
                            conv,
                            act,
                            norm,
                            bias,
                            use_stochastic,
                            epsilon,
                            reduce_ratios[i],
                            n=HW,
                            drop_path=dpr[idx],
                            relative_pos=True),
                        FFN(channels[i],
                            channels[i] * 4,
                            act=act,
                            drop_path=dpr[idx]))
                ]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(
            nn.Conv2d(channels[-1], 1024, 1, bias=True), nn.BatchNorm2d(1024),
            act_layer(act), nn.Dropout(dropout),
            nn.Conv2d(1024, n_classes, 1, bias=True))

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed

        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = self.prediction(x).squeeze(-1).squeeze(-1)
        return (x, )

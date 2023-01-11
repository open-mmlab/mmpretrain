# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import Sequential

from mmcls.registry import MODELS
from .cls_head import ClsHead


@MODELS.register_module()
class VigClsHead(ClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 hidden_dim: Optional[int] = 1024,
                 act_cfg: dict = dict(type='GELU'),
                 dropout=0,
                 **kwargs):
        super(VigClsHead, self).__init__(**kwargs)

        self.classifier = Sequential(
            build_conv_layer(None, in_channels, hidden_dim, 1, bias=True),
            build_norm_layer(dict(type='BN'), hidden_dim)[1],
            build_activation_layer(act_cfg), nn.Dropout(dropout),
            build_conv_layer(None, hidden_dim, num_classes, 1, bias=True))

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a stage_blocks stage. In ``VigClsHead``, we just obtain the
        feature of the last stage.
        """

        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.classifier(pre_logits)
        return cls_score.squeeze(-1).squeeze(-1)

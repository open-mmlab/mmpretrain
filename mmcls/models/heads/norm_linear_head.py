# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.data import ClsDataSample
from mmcls.models.heads import ClsHead
from mmcls.registry import MODELS


class NormLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 feature_norm: bool, weight_norm: bool):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_norm = weight_norm
        self.feature_norm = feature_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_norm:
            input = F.normalize(input)
        if self.weight_norm:
            weight = F.normalize(self.weight)
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)


@MODELS.register_module()
class NormLinearClsHead(ClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 feature_norm: bool,
                 weight_norm: bool,
                 bias: bool,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(NormLinearClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)

        assert hasattr('s', self.compute_loss),\
            'NormLinearClsHead.compute_loss should have `s` like ArcFaceLoss.'

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = NormLinear(self.in_channels, self.num_classes, bias,
                             feature_norm, weight_norm)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The NormLinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)
        cls_score = self.fc(pre_logits)
        return cls_score

    def predict(
            self,
            feats: Tuple[torch.Tensor],
            data_samples: List[ClsDataSample] = None) -> List[ClsDataSample]:
        """Test without augmentation."""
        # The part can be traced by torch.fx
        cls_score = self(feats)
        cls_score = cls_score * self.compute_loss.s

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

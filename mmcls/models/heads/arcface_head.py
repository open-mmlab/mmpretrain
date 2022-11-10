# Copyright (c) OpenMMLab. All rights reserved.
import math
import pickle
from ctypes import Union
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.fileio import list_from_file
from mmengine.runner import autocast
from mmengine.structures import LabelData

from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from .cls_head import ClsHead


class NormProduct(nn.Linear):
    """An enhanced linear layer with k clustering centers to calculate product
    between normalized input and linear weight.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample
        k (int): The number of clustering centers. Defaults to 3.
        bias (bool): Whether there is bias. If set to ``False``, the
            layer will not learn an additive bias. Defaults to ``True``.
        feature_norm (bool): Whether to normalize the input feature.
            Defaults to ``True``.
        weight_norm (bool):Whether to normalize the weight.
            Defaults to ``True``.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 k=1,
                 bias: bool = False,
                 feature_norm: bool = True,
                 weight_norm: bool = True):

        super().__init__(in_features, out_features * k, bias=bias)
        self.weight_norm = weight_norm
        self.feature_norm = feature_norm
        self.out_features = out_features
        self.k = k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_norm:
            input = F.normalize(input)
        if self.weight_norm:
            weight = F.normalize(self.weight)
        else:
            weight = self.weight
        cosine_all = F.linear(input, weight, self.bias)
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


@MODELS.register_module()
class ArcFaceClsHead(ClsHead):
    """ArcFace classifier head.

    A PyTorch implementation of paper `ArcFace: Additive Angular Margin Loss
     for Deep Face Recognition <https://arxiv.org/abs/1801.07698>`_ and
    `Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web
    Faces <https://link.springer.com/chapter/10.1007/978-3-030-58621-8_43>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_subcenters (int): Number of subcenters. Defaults to 1.
        scale (float): Scale factor of output logit. Defaults to 30.0.
        margin (float): The penalty margin. Could be the fllowing formats:

            - float: The penalty margin.
            - Sequence: The category-based penalty margins.
            - str: a '.txt' or '.pkl' file that contains the penalty margins.

            Defaults to 0.5.
        easy_margin (bool): Avoid theta + m >= PI. Defaults to False.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_subcenters: int = 1,
                 scale: float = 30.0,
                 margin: Union[float, Sequence, str] = 0.50,
                 easy_margin: bool = False,
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[dict] = None):

        super(ArcFaceClsHead, self).__init__(init_cfg=init_cfg)
        self.loss_module = MODELS.build(loss)

        assert num_subcenters >= 1 and num_classes >= 0
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters
        self.scale = scale
        self.easy_margin = easy_margin
        self.with_adaptive_margin = not isinstance(margin, float)

        self.norm_product = NormProduct(in_channels, num_classes,
                                        num_subcenters)

        if not self.with_adaptive_margin:
            self.margin = margin
            self.threshold = math.cos(math.pi - margin)
            self.mm = math.sin(math.pi - margin) * margin  # the fixd penalty

        if isinstance(margin, float):
            margins = [margin] * num_classes
        elif isinstance(margin, str) and margin.endswith('.txt'):
            margins = [float(item) for item in list_from_file(margin)]
        elif isinstance(margin, str) and margin.endswith('.pkl'):
            with open(margin, 'r') as pkl_file:
                margins = pickle.load(pkl_file)
        else:
            assert isinstance(margin, Sequence), (
                'the attr `margin` in ``ArcFaceClsHead`` should be float,'
                ' Sequence, or .txt, .pkl file contain sequence data.')

        assert len(margins) == num_classes, \
            'The size of margin must be equal with num_classes.'
        margins = torch.tensor(margins).float()
        self.register_buffer('margins', margins)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ArcFaceHead``, we just obtain the
        feature of the last stage.
        """
        # The ArcFaceHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self,
                feats: Tuple[torch.Tensor],
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The forward process."""

        assert target.dim() == 1 or (
            target.dim() == 2 and target.shape[1] == 1), \
            '``ArcFaceClsHead`` only support index format target.'

        with autocast(enabled=False):
            pre_logits = self.pre_logits(feats)

            # cos(theta_yj) = (x/||x||) * (W/||W||)
            cosine = self.norm_product(pre_logits)

            if target is None:
                return self.scale * cosine

            if self.with_adaptive_margin:
                margin = self.margins[target].unsqueeze(1)
                mm = torch.sin(math.pi - margin) * margin
                threshold = torch.cos(math.pi - margin)
            else:
                margin = self.margin
                mm = self.threshold
                threshold = self.threshold

            phi = torch.cos(torch.acos(cosine) + margin)

            if self.easy_margin:
                # when cosine>0, choose phi
                # when cosine<=0, choose cosine
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                # when cos>th, choose phi
                # when cos<=th, choose cosine-mm
                phi = torch.where(cosine > threshold, phi, cosine - mm)

            target = LabelData.label_to_onehot(target, cosine.size())
            output = (target * phi) + ((1.0 - target) * cosine)

        return output * self.scale

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[ClsDataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        target = torch.cat([i.gt_label.label for i in data_samples])

        # The part can be traced by torch.fx
        cls_score = self(feats, target)

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        return losses

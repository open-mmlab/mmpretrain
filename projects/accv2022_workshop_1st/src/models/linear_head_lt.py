# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import mmengine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.heads import LinearClsHead
from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample


def compute_adjustment(ann_file, tro=1):
    """compute the base probabilities."""
    lines = mmengine.list_from_file(ann_file)
    samples = [x.strip().rsplit(' ', 1) for x in lines]

    label_freq = {}
    for filename, target in samples:
        label_freq[target] = label_freq.get(target, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array**tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    return adjustments


@MODELS.register_module(force=True)
class LinearClsHeadWithAdjustment(LinearClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 mod_loss=False,
                 adjustments=None,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        self.fc = nn.Linear(self.in_channels, self.num_classes)

        if isinstance(adjustments, str):
            adjustments = compute_adjustment(adjustments)

        if adjustments is not None:
            self.register_buffer('adjustments', adjustments)
            print(f'Using Long Tail Adjustments {len(self.adjustments)}.')
        else:
            self.adjustments = adjustments

        self.mod_loss = mod_loss

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
        # The part can be traced by torch.fx
        cls_score = self(feats)
        if self.adjustments is not None:
            cls_score += self.adjustments

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def predict(
            self,
            feats: Tuple[torch.Tensor],
            data_samples: List[ClsDataSample] = None) -> List[ClsDataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[ClsDataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        if self.adjustments is not None and self.mod_loss:
            cls_score = cls_score + self.adjustments
        elif self.adjustments is not None and self.mod_loss is False:
            cls_score = cls_score - self.adjustments

        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        if data_samples is not None:
            for data_sample, score, label in zip(data_samples, pred_scores,
                                                 pred_labels):
                data_sample.set_pred_score(score).set_pred_label(label)
        else:
            data_samples = []
            for score, label in zip(pred_scores, pred_labels):
                data_samples.append(ClsDataSample().set_pred_score(
                    score).set_pred_label(label))

        return data_samples

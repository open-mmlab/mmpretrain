# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from mmengine.logging import MessageHub

from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from .linear_head import LinearClsHead


@MODELS.register_module()
class LogitAdjustLinearClsHead(LinearClsHead):
    """logit adjustment Linear classifier head.

    A PyTorch implementation of paper `Long-tail learning via logit
    adjustment <https://arxiv.org/abs/2007.07314>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        enable_posthoc_adjustment (bool): Whether to enable post-hoc
            logit adjustment. Defaults to Flase.
        enable_loss_adjustment (bool): Whether to enable loss logit
            adjustment. Defaults to False.
        tau (float) : Tau parameter for logit adjustment. Defaults to 1.0.
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
                 enable_posthoc_adjustment: bool = False,
                 enable_loss_adjustment: bool = False,
                 tau: float = 1.0,
                 **kwargs):
        super(LogitAdjustLinearClsHead, self).__init__(**kwargs)

        if enable_posthoc_adjustment and enable_loss_adjustment:
            raise ValueError(
                'Only one of the `enable_post_adjustment` '
                'and `enable_loss_adjustment` can be set to True.')
        if not enable_posthoc_adjustment and not enable_loss_adjustment:
            raise ValueError(
                'One of the `enable_post_adjustment` '
                'and `enable_loss_adjustment` must be set to True.')

        self.enable_posthoc_adjustment = enable_posthoc_adjustment
        self.enable_loss_adjustment = enable_loss_adjustment
        self.tau = tau
        self._adjustments = None

    def get_adjustments(self) -> torch.Tensor:
        """get adjustments."""
        if self._adjustments is None:
            try:
                # try to get dataset gt_labels from Messagehub.
                gt_labels = MessageHub.get_current_instance().get_info(
                    'gt_labels')
            except KeyError:
                raise RuntimeError(
                    "Please set ``dict(type='PushDataInfoToMessageHubHook', "
                    "keys=['gt_labels'])`` in custom_hooks config to push"
                    ' gt_labels info to MessageHub.')

            label_count = np.bincount(gt_labels)
            freq = label_count / np.sum(label_count)

            self._adjustments = torch.from_numpy(
                np.log(freq**self.tau + 1e-12))
            self._adjustments.to(self.fc.weight.device)
        return self._adjustments

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

        # if enable loss_adjustment, logits plus adjustments in ``loss()``.
        if self.enable_loss_adjustment:
            cls_score += self.get_adjustments().to(cls_score.device)

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

        # if enable post-hoc, logits minus adjustments in ``predict()``.
        if self.enable_posthoc_adjustment:
            cls_score -= self.get_adjustments().to(cls_score.device)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

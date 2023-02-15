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
        enable_post_hoc_adjustment (bool): Whether to enable post-hoc
            logit adjustment. Defaults to True.
        enable_loss_adjustment (bool): Whether to enable post logit
            adjustment. Defaults to False.
        pow (float) : The pow used to calculate adjustment from
            label frequency. Defaults to 1.0.
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
                 *args,
                 enable_post_hoc_adjustment: bool = True,
                 enable_loss_adjustment: bool = False,
                 pow: float = 1.0,
                 **kwargs):
        super(LinearClsHead, self).__init__(*args, **kwargs)

        if enable_post_hoc_adjustment and enable_loss_adjustment:
            raise ValueError(
                'Only one of the `enable_post_adjustment` '
                'and `enable_loss_adjustment` can be set to True.')
        if not enable_post_hoc_adjustment and not enable_loss_adjustment:
            raise ValueError(
                'One of the `enable_post_adjustment` '
                'and `enable_loss_adjustment` must be set to True.')

        self.enable_post_hoc_adjustment = enable_post_hoc_adjustment
        self.enable_loss_adjustment = enable_loss_adjustment
        self.pow = pow
        self._adjustments = None

    @property
    def adjustments(self, ) -> torch.Tensor:
        """get adjustments."""
        if self._adjustments is None:
            try:
                # try to get dataset gt_labels from Messagehub.
                gt_labels = MessageHub.get_current_instance().get_info(
                    'dataset_gt_labels')
            except KeyError:
                raise RuntimeError(
                    'Please set ```train_dataloader=dict(...,dataset=dict('
                    '..., global_gt_labels=True, ...,))``` in config to '
                    "send  'dataset_gt_labels' info to MessageHub.")

            label_count = np.bincount(gt_labels)
            freq = label_count / np.sum(label_count)

            self._adjustments = torch.from_numpy(
                np.log(freq**self.pow + 1e-12))
            self._adjustments.to(self.device)

        return self._adjustments

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)

        # if enable loss adjustment, logits add adjustments in
        # both ``loss`` and ``predict`` process.
        if self.enable_loss_adjustment:
            cls_score += self.adjustments

        return cls_score

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

        # if enable post-hoc, logits minus adjustments in ``predict``.
        if self.enable_post_hoc_adjustment:
            cls_score -= self.adjustments

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

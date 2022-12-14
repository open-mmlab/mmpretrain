# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmengine.structures import LabelData

from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from .base_head import BaseHead


@MODELS.register_module()
class MultiLabelClsHead(BaseHead):
    """Classification head for multilabel task.

    Args:
        loss (dict): Config of classification loss. Defaults to
            dict(type='CrossEntropyLoss', use_sigmoid=True).
        thr (float, optional): Predictions with scores under the thresholds
            are considered as negative. Defaults to None.
        topk (int, optional): Predictions with the k-th highest scores are
            considered as positive. Defaults to None.
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to None.

    Notes:
        If both ``thr`` and ``topk`` are set, use ``thr` to determine
        positive predictions. If neither is set, use ``thr=0.5`` as
        default.
    """

    def __init__(self,
                 loss: Dict = dict(type='CrossEntropyLoss', use_sigmoid=True),
                 thr: Optional[float] = None,
                 topk: Optional[int] = None,
                 init_cfg: Optional[dict] = None):
        super(MultiLabelClsHead, self).__init__(init_cfg=init_cfg)

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss

        if thr is None and topk is None:
            thr = 0.5

        self.thr = thr
        self.topk = topk

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``MultiLabelClsHead``, we just obtain
        the feature of the last stage.
        """
        # The MultiLabelClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The MultiLabelClsHead doesn't have the final classification head,
        # just return the unpacked inputs.
        return pre_logits

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

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[ClsDataSample], **kwargs):
        """Unpack data samples and compute loss."""
        num_classes = cls_score.size()[-1]
        # Unpack data samples and pack targets
        if 'score' in data_samples[0].gt_label:
            target = torch.stack(
                [i.gt_label.score.float() for i in data_samples])
        else:
            target = torch.stack([
                LabelData.label_to_onehot(i.gt_label.label,
                                          num_classes).float()
                for i in data_samples
            ])

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

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

    def _get_predictions(self, cls_score: torch.Tensor,
                         data_samples: List[ClsDataSample]):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = torch.sigmoid(cls_score)

        if data_samples is None:
            data_samples = [ClsDataSample() for _ in range(cls_score.size(0))]

        for data_sample, score in zip(data_samples, pred_scores):
            if self.thr is not None:
                # a label is predicted positive if larger than thr
                label = torch.where(score >= self.thr)[0]
            else:
                # top-k labels will be predicted positive for any example
                _, label = score.topk(self.topk)
            data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import HEADS, build_loss
from ..utils import is_tracing
from .base_head import BaseHead


@HEADS.register_module()
class MultiLabelClsHead(BaseHead):
    """Classification head for multilabel task.

    Args:
        loss (dict): Config of classification loss.
    """

    def __init__(self,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=None):
        super(MultiLabelClsHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)

        self.compute_loss = build_loss(loss)

    def loss(self, cls_score, gt_label):
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()

        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        loss = self.compute_loss(cls_score, _gt_label, avg_factor=num_samples)
        losses['loss'] = loss
        return losses

    def forward_train(self, cls_score, gt_label):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        gt_label = gt_label.type_as(cls_score)
        losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        if isinstance(x, list):
            x = sum(x) / float(len(x))
        pred = F.sigmoid(x) if x is not None else None

        return self.post_process(pred)

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

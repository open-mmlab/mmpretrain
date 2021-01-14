import torch
import torch.nn.functional as F

from ..builder import HEADS, build_loss
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
                     loss_weight=1.0)):
        super(MultiLabelClsHead, self).__init__()

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
        gt_label = gt_label.type_as(cls_score)
        losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, cls_score):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.sigmoid(cls_score) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

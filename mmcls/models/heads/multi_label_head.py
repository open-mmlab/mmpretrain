# Copyright (c) OpenMMLab. All rights reserved.
import torch

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

    def forward_train(self, cls_score, gt_label, **kwargs):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        gt_label = gt_label.type_as(cls_score)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]

        from mmcls.utils import get_root_logger
        logger = get_root_logger()
        logger.warning(
            'The input of MultiLabelClsHead should be already logits. '
            'Please modify the backbone if you want to get pre-logits feature.'
        )
        return x

    def simple_test(self, x, sigmoid=True, post_process=True):
        """Inference without augmentation.

        Args:
            cls_score (tuple[Tensor]): The input classification score logits.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            sigmoid (bool): Whether to sigmoid the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        if isinstance(x, tuple):
            x = x[-1]

        if sigmoid:
            pred = torch.sigmoid(x) if x is not None else None
        else:
            pred = x

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

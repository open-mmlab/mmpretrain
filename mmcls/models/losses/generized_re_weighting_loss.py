# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.registry import MODELS
from .utils import weight_reduce_loss


def generalized_re_weighting_loss(pred,
                                  label,
                                  weight=None,
                                  reduction='mean',
                                  avg_factor=None,
                                  class_weight=None):
    """Calculate the Generalized Re-weight Loss, introduced in Distribution
    Alignment: A Unified Framework for Long-tail Visual Recognition
    https://arxiv.org/abs/2103.16370.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@MODELS.register_module()
class GRWLoss(nn.Module):
    """GRWLoss.

    Args:
        exp_scale (float): Scale hyper-parameter in GRWloss.
            Defaults to 1.0.
        reduction (str): The method used to reduce the loss into
            a scalar. Options are "none" and "mean". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        samples_per_cls (list[int]): The number of samples per class.
            Default: [].
        num_classes (int): Number of classes. Defaults to 1000.
        cls_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        frequency_from (str): The used dataset frequency file. Defaults None.
    """

    def __init__(self,
                 exp_scale=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 samples_per_cls=[],
                 num_classes=1000,
                 cls_weight=None,
                 frequency_from=None):

        super(GRWLoss, self).__init__()
        self.exp_scale = exp_scale
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.samples_per_cls = samples_per_cls
        self.cls_weight = cls_weight
        if frequency_from is not None:
            import json
            with open(frequency_from, encoding='utf-8') as a:
                result = json.load(a)
                print(result['train_shots'])
                self.samples_per_cls = result['train_shots']
                cls_weight = (
                    torch.tensor(self.samples_per_cls).float()**self.exp_scale)
                self.cls_weight = (
                    (1 / cls_weight) /
                    (1 / cls_weight).sum()).float() * num_classes

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        r"""Generalized re-weighting loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \*).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*), N or (N,1).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        cls_weight = self.cls_weight.to(target.device)
        loss_cls = self.loss_weight * generalized_re_weighting_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            class_weight=cls_weight)
        return loss_cls

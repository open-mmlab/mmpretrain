import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    """Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), where C is the
            number of classes.
        label (torch.Tensor): The learning label with shape (N, C), where C is
            the number of classes.
        weight (torch.Tensor, optional): Weight of loss. Defaults to None.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert pred.dim() == label.dim()

    loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, reduction='mean', loss_weight=1.0):
        """Cross entropy loss

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
            loss_weight (float, optional):  Weight of the loss.
                Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

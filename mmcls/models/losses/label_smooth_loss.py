import numpy as np

from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss
from .utils import convert_to_one_hot


@LOSSES.register_module()
class LabelSmoothLoss(CrossEntropyLoss):
    """Intializer for the label smoothed cross entropy loss.

    This decreases gap between output scores and encourages generalization.
    Labels provided to forward can be one-hot like vectors (NxC) or class
    indices (Nx1).
    This normalizes the labels to a sum of 1 based on the total count of
    positive targets for a given sample before applying label smoothing.

    Args:
        label_smooth_val (float): Value to be added to each target entry
        num_classes (int, optional): Number of classes. Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 label_smooth_val,
                 num_classes=None,
                 reduction='mean',
                 loss_weight=1.0):
        super(LabelSmoothLoss, self).__init__(
            use_sigmoid=False,
            use_soft=True,
            reduction=reduction,
            loss_weight=loss_weight)
        self._label_smooth_val = label_smooth_val
        self.num_classes = num_classes
        self._eps = np.finfo(np.float32).eps

    def generate_one_hot_like_label(self, label):
        """This function takes one-hot or index label vectors and computes one-
        hot like label vectors (float)"""
        label_shape_list = list(label.size())
        # check if targets are inputted as class integers
        if len(label_shape_list) == 1 or (len(label_shape_list) == 2
                                          and label_shape_list[1] == 1):
            label = convert_to_one_hot(label.view(-1, 1), self.num_classes)
        return label.float()

    def smooth_label(self, one_hot_like_label):
        """This function takes one-hot like target vectors and computes
        smoothed target vectors (normalized) according to the loss's smoothing
        parameter."""
        assert self.num_classes > 0
        one_hot_like_label /= self._eps + one_hot_like_label.sum(
            dim=1, keepdim=True)
        smoothed_targets = one_hot_like_label + (
            self._label_smooth_val / self.num_classes)
        smoothed_targets /= self._eps + smoothed_targets.sum(
            dim=1, keepdim=True)

        return smoothed_targets

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if self.num_classes is not None:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'
        else:
            self.num_classes = cls_score.shape[1]
        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        assert (
            one_hot_like_label.shape == cls_score.shape
        ), f'LabelSmoothingCrossEntropyLoss requires output and target ' \
           f'to be same shape, but got output.shape: {cls_score.shape}' \
           f'and target.shape: {one_hot_like_label.shape}'
        smoothed_label = self.smooth_label(
            one_hot_like_label=one_hot_like_label)
        return super(LabelSmoothLoss, self).forward(
            cls_score,
            smoothed_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)

import numpy as np

from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss
from .utils import convert_to_one_hot


@LOSSES.register_module()
class LabelSmoothCrossEntropyLoss(CrossEntropyLoss):
    """Intializer for the label smoothed cross entropy loss.
        This decreases gap between output scores and encourages generalization.
        Targets provided to forward can be one-hot like vectors (NxC) or class
        indices (Nx1).
        This normalizes the targets to a sum of 1 based on the total count of
        positive targets for a given sample before applying label smoothing.

        Args:
            reduction: specifies reduction to apply to the output
            smoothing_param: value to be added to each target entry
        """

    def __init__(self,
                 reduction='mean',
                 smoothing_param=None,
                 loss_weight=1.0):
        super(LabelSmoothCrossEntropyLoss, self).__init__(
            use_sigmoid=False,
            use_soft=True,
            reduction=reduction,
            loss_weight=loss_weight)
        self._smoothing_param = smoothing_param
        self._eps = np.finfo(np.float32).eps

    def generate_one_hot_like_label(self, label, classes):
        """
        This function takes one-hot or index label vectors and computes
        one-hot like label vectors (float)
        """
        label_shape_list = list(label.size())
        # check if targets are inputted as class integers
        if len(label_shape_list) == 1 or (len(label_shape_list) == 2
                                          and label_shape_list[1] == 1):
            label = convert_to_one_hot(label.view(-1, 1), classes)
        return label.float()

    def smooth_label(self, one_hot_like_label, classes):
        """
        This function takes one-hot like target vectors and
        computes smoothed target vectors (normalized)
        according to the loss's smoothing parameter
        """
        assert classes > 0
        one_hot_like_label /= self._eps + one_hot_like_label.sum(
            dim=1, keepdim=True)
        smoothed_targets = one_hot_like_label + (
            self._smoothing_param / classes)
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
        one_hot_like_label = self.generate_one_hot_like_label(
            label=label, classes=cls_score.shape[1])
        assert (
            one_hot_like_label.shape == cls_score.shape
        ), 'LabelSmoothingCrossEntropyLoss requires output and ' \
           'target to be same size'
        smoothed_label = self.smooth_label(
            one_hot_like_label=one_hot_like_label, classes=cls_score.shape[1])
        return super(LabelSmoothCrossEntropyLoss, self).forward(
            cls_score,
            smoothed_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)

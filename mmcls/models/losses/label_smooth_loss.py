# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmcls.registry import MODELS
from .cross_entropy_loss import BinaryCrossEntropyLoss, CrossEntropyLoss
from .utils import convert_to_one_hot


@MODELS.register_module()
class LabelSmoothLoss(nn.Module):
    r"""Label smoothed (binary) cross entropy loss.

    Refers to `Rethinking the Inception Architecture for Computer Vision
    <https://arxiv.org/abs/1512.00567>`_

    This decreases gap between output scores and encourages generalization.
    Labels provided to forward can be one-hot like vectors (NxC) or class
    indices (Nx1).
    And this accepts linear combination of one-hot like labels from mixup or
    cutmix except multi-label task.

    Args:
        label_smooth_val (float): The degree of label smoothing.
        num_classes (int, optional): Number of classes. Defaults to None.
        mode (str): Refers to notes, Options are 'original', 'classy_vision',
            'multi_label'. Defaults to 'original'.
        use_sigmoid (bool, optional): If True, do sigmoid before calculating
            cross entropy (i.e. BCE). If False, do softmax before calculating
            cross entropy. Defaults to None, means True for 'multi_label' mode,
            and False for other mode.
        **kwargs: If ``use_sigmoid=False``, accepts other keyword arguments
            of :class:`CrossEntropyLoss`. If ``use_sigmoid=True``, accepts
            other keyword arguments of :class:`BinaryCrossEntropyLoss`.

    Notes:
        - if the mode is **"original"**, this will use the same label smooth
          method as the original paper as:

          .. math::
              (1-\epsilon)\delta_{k, y} + \frac{\epsilon}{K}

          where :math:`\epsilon` is the ``label_smooth_val``, :math:`K` is the
          ``num_classes`` and :math:`\delta_{k, y}` is Dirac delta, which
          equals 1 for :math:`k=y` and 0 otherwise.

        - if the mode is **"classy_vision"**, this will use the same label
          smooth method as the facebookresearch/ClassyVision repo as:

          .. math::
              \frac{\delta_{k, y} + \epsilon/K}{1+\epsilon}

        - if the mode is **"multi_label"**, this will accept labels from
          multi-label task and smoothing them as:

          .. math::
              (1-2\epsilon)\delta_{k, y} + \epsilon
    """

    def __init__(self,
                 label_smooth_val,
                 num_classes=None,
                 use_sigmoid=None,
                 mode='original',
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes

        assert (isinstance(label_smooth_val, float)
                and 0 <= label_smooth_val < 1), \
            f'LabelSmoothLoss accepts a float label_smooth_val ' \
            f'over [0, 1), but gets {label_smooth_val}'
        self.label_smooth_val = label_smooth_val

        accept_mode = {'original', 'classy_vision', 'multi_label'}
        assert mode in accept_mode, \
            f'LabelSmoothLoss supports mode {accept_mode}, but gets {mode}.'
        self.mode = mode

        self._eps = label_smooth_val
        if mode == 'classy_vision':
            self._eps = label_smooth_val / (1 + label_smooth_val)

        if mode == 'multi_label':
            self.smooth_label = self.multilabel_smooth_label
            use_sigmoid = True if use_sigmoid is None else use_sigmoid
        else:
            self.smooth_label = self.original_smooth_label
            use_sigmoid = False if use_sigmoid is None else use_sigmoid

        if use_sigmoid:
            self.ce = BinaryCrossEntropyLoss(**kwargs)
        else:
            self.ce = CrossEntropyLoss(use_soft=True, **kwargs)

    def generate_one_hot_like_label(self, label):
        """This function takes one-hot or index label vectors and computes one-
        hot like label vectors (float)"""
        # check if targets are inputted as class integers
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = convert_to_one_hot(label.view(-1, 1), self.num_classes)
        return label.float()

    def original_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = one_hot_like_label * (1 - self._eps)
        smooth_label += self._eps / self.num_classes
        return smooth_label

    def multilabel_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = torch.full_like(one_hot_like_label, self._eps)
        smooth_label.masked_fill_(one_hot_like_label > 0, 1 - self._eps)
        return smooth_label

    def forward(self, cls_score, label, **kwargs):
        r"""Forward label smooth loss.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, \*).
            label (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*).
            **kwargs: If ``self.use_sigmoid=False``, accepts other keyword
                arguments of :meth:`CrossEntropyLoss.forward`. If
                ``self.use_sigmoid=True``, accepts other keyword arguments of
                :meth:`BinaryCrossEntropyLoss.forward`.

        Returns:
            torch.Tensor: Loss.
        """
        if self.num_classes is not None:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'
        else:
            self.num_classes = cls_score.shape[1]

        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        assert one_hot_like_label.shape == cls_score.shape, \
            f'LabelSmoothLoss requires output and target ' \
            f'to be same shape, but got output.shape: {cls_score.shape} ' \
            f'and target.shape: {one_hot_like_label.shape}'

        smoothed_label = self.smooth_label(one_hot_like_label)
        return self.ce.forward(cls_score, smoothed_label, **kwargs)

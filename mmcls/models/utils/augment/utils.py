# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F


def one_hot_encoding(gt, num_classes):
    """Change gt_label to one_hot encoding.

    If the shape has 2 or more
    dimensions, return it without encoding.
    Args:
        gt (Tensor): The gt label with shape (N,) or shape (N, */).
        num_classes (int): The number of classes.
    Return:
        Tensor: One hot gt label.
    """
    if gt.ndim == 1:
        # multi-class classification
        return F.one_hot(gt, num_classes=num_classes)
    else:
        # binary classification
        # example. [[0], [1], [1]]
        # multi-label classification
        # example. [[0, 1, 1], [1, 0, 0], [1, 1, 1]]
        return gt

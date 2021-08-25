# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def average_precision(pred, target):
    r"""Calculate the average precision for a single class.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        pred (np.ndarray): The model prediction with shape (N, ).
        target (np.ndarray): The target of each prediction with shape (N, ).

    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    pos_inds = sort_target == 1
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap


def mAP(pred, target):
    """Calculate the mean average precision with respect of classes.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.

    Returns:
        float: A single float as mAP value.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    num_classes = pred.shape[1]
    ap = np.zeros(num_classes)
    for k in range(num_classes):
        ap[k] = average_precision(pred[:, k], target[:, k])
    mean_ap = ap.mean() * 100.0
    return mean_ap

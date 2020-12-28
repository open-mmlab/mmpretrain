import warnings

import numpy as np
import torch


def average_performance(pred, target, thrs=None, k=None):
    """Calculate CP, CR, CF1, OP, OR, OF1, where C stands for per-class
        average, O stands for overall average, P stands for precision, R
        stands for recall and F1 stands for F1-score

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction.
        target (torch.Tensor | np.ndarray): The target of each prediction, in
            which 1 stands for positive examples and both -1 and 0 stand for
            negative examples.
        thrs (float): The confidence threshold. Defaults to None.
        k (int): Top-k performance. Note that if thrs and k are both given, k
            will be ignored. Defaults to None.

    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1)
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.numpy()
        target = target.numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')
    if thrs is None and k is None:
        thrs = 0.5
        warnings.warn('Neither thrs nor k is given, set thrs as 0.5 by '
                      'default.')
    elif thrs is not None and k is not None:
        warnings.warn('Both thrs and k are given, use threshold in favor of '
                      'top-k.')

    assert pred.shape == target.shape

    eps = np.finfo(np.float32).eps
    target[target == -1] = 0
    if thrs is not None:
        # a label is predicted positive if the confidence is no lower than thrs
        p_inds = pred >= thrs

    else:
        # top-k labels will be predicted positive for any example
        sort_inds = np.argsort(-pred, axis=1)
        sort_inds_ = sort_inds[:, :k]
        inds = np.indices(sort_inds_.shape)
        p_inds = np.zeros_like(pred)
        p_inds[inds[0], sort_inds_] = 1

    tp = (p_inds * target) == 1
    fp = (p_inds * (1 - target)) == 1
    fn = ((1 - p_inds) * target) == 1

    precision_class = tp.sum(0) / (tp.sum(0) + fp.sum(0) + eps)
    recall_class = tp.sum(0) / (tp.sum(0) + fn.sum(0) + eps)
    CP = precision_class.mean() * 100.0
    CR = recall_class.mean() * 100.0
    CF1 = 2 * CP * CR / (CP + CR + eps)
    OP = tp.sum() / (tp.sum() + fp.sum() + eps) * 100.0
    OR = tp.sum() / (tp.sum() + fn.sum() + eps) * 100.0
    OF1 = 2 * OP * OR / (OP + OR + eps)
    return CP, CR, CF1, OP, OR, OF1

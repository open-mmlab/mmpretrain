import numpy as np
import torch


def average_precision(pred, target):
    """ Calculate the average precision for a single class

    Args:
        pred (np.ndarray): The model prediction.
        target (np.ndarray): The target of each prediction.

    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    p_inds = sort_target == 1
    tp = np.cumsum(p_inds)
    total_p = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != 0
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(p_inds)] = 0
    precision = tp / (pn + eps)
    ap = np.sum(precision) / (total_p + eps)
    return ap


def mAP(pred, target, difficult_examples=True):
    """ Calculate the mean average precision with respect of classes

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction.
        target (torch.Tensor | np.ndarray): The target of each prediction. If
            difficult_examples is set as True, 1 stands for positive examples,
            0 stands for difficult examples and -1 stands for negative
            examples. Otherwise, 1 stands for positive examples and 0 stands
            for negative examples.
        difficult_examples (bool): Whether dataset contains difficult_examples.
            Defaults to True.

    Returns:
        float: A single float as mAP value.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.numpy()
        target = target.numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == target.shape
    num_classes = pred.shape[1]
    if not difficult_examples:
        target[target == 0] = -1
    ap = np.zeros(num_classes)
    for k in range(num_classes):
        ap[k] = average_precision(pred[:, k], target[:, k])
    mean_ap = ap.mean() * 100.0
    return mean_ap

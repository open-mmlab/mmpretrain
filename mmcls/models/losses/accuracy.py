import numpy as np
import torch
import torch.nn as nn


def accuracy_numpy(pred, target, topk):
    res = []
    maxk = max(topk)
    num = pred.shape[0]
    pred_label = pred.argsort(axis=1)[:, -maxk:][:, ::-1]

    for k in topk:
        correct_k = np.logical_or.reduce(
            pred_label[:, :k] == target.reshape(-1, 1), axis=1)
        res.append(correct_k.sum() * 100. / num)
    return res


def accuracy_torch(pred, target, topk=1):
    res = []
    maxk = max(topk)
    num = pred.size(0)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100. / num))
    return res


def accuracy(pred, target, topk=1):
    """Calculate accuracy according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        res = accuracy_torch(pred, target, topk)
    elif isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        res = accuracy_numpy(pred, target, topk)
    else:
        raise TypeError('pred and target should both be'
                        'torch.Tensor or np.ndarray')

    return res[0] if return_single else res


class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        """Module to calculate the accuracy

        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        """Forward function to calculate accuracy

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk)

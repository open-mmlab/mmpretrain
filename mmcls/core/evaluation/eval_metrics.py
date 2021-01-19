import numpy as np
import torch


def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.

    Returns:
        torch.Tensor: Confusion matrix with shape (C, C), where C is the number
             of classes.
    """
    if isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        pred = torch.from_numpy(pred)
        target = torch.from_numpy(target)
    elif not (isinstance(pred, torch.Tensor)
              and isinstance(target, torch.Tensor)):
        raise TypeError('pred and target should both be'
                        'torch.Tensor or np.ndarray')
    _, pred_label = pred.topk(1, dim=1)
    num_classes = pred.size(1)
    pred_label = pred_label.view(-1)
    target_label = target.view(-1)
    assert len(pred_label) == len(target_label)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for t, p in zip(target_label, pred_label):
            confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def precision(pred, target, average='macro'):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.
        average (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. Defaults to 'macro'.

    Returns:
        np.array: Precision value with shape determined by average.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(0), min=1) * 100
        if average == 'macro':
            res = res.mean().numpy()
        elif average == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average}.')
    return res


def recall(pred, target, average='macro'):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.
        average (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. Defaults to 'macro'.

    Returns:
        np.array: Recall value with shape determined by average.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(1), min=1) * 100
        if average == 'macro':
            res = res.mean().numpy()
        elif average == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average}.')
    return res


def f1_score(pred, target, average='macro'):
    """Calculate F1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.
        average (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. Defaults to 'macro'.

    Returns:
        np.array: F1 score with shape determined by average.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        precision = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(1), min=1)
        recall = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(0), min=1)
        res = 2 * precision * recall / torch.clamp(
            precision + recall, min=1e-20) * 100
        res = torch.where(torch.isnan(res), torch.full_like(res, 0), res)
        if average == 'macro':
            res = res.mean().numpy()
        elif average == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average}.')
    return res


def support(pred, target, average='macro'):
    """Calculate the total number of occurrences of each label according to
        the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.
        average (str): The type of reduction performed on the result.
            Options are 'macro' and 'none'. 'macro' gives the sum and 'none'
            gives class-wise results. Defaults to 'macro'.

    Returns:
        np.array: Support with shape determined by average.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average == 'macro':
            res = res.sum().numpy()
        elif average == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average}.')
    return res

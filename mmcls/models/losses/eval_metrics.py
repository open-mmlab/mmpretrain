import numpy as np
import torch


def calculate_confusion_matrix_torch(pred, target):
    _, pred_label = pred.topk(1, dim=1)
    num_classes = pred.size(1)
    pred_label = pred_label.view(-1)
    target_label = target.view(-1)
    assert len(pred_label) == len(target_label)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        confusion_matrix[target_label.long(), pred_label.long()] += 1
    return confusion_matrix


def calculate_confusion_matrix_numpy(pred, target):
    pred_label = pred.argsort(axis=1)[:, -1]
    num_classes = pred.shape[1]
    pred_label = pred_label.reshape(-1)
    target_label = target.reshape(-1)
    assert len(pred_label) == len(target_label)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        confusion_matrix[target_label, pred_label] += 1
    return confusion_matrix


def precision(pred, target):
    """Calculate macro-averaged precision according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.

    Returns:
        float: The function will return a single float as precision.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        confusion_matrix = calculate_confusion_matrix_torch(pred, target)
    elif isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        confusion_matrix = calculate_confusion_matrix_numpy(pred, target)
    else:
        raise TypeError('pred and target should both be'
                        'torch.Tensor or np.ndarray')
    with torch.no_grad():
        res = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(1), min=1)
        res = res.mean().item() * 100
    return res


def recall(pred, target):
    """Calculate macro-averaged recall according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.

    Returns:
        float: The function will return a single float as recall.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        confusion_matrix = calculate_confusion_matrix_torch(pred, target)
    elif isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        confusion_matrix = calculate_confusion_matrix_numpy(pred, target)
    else:
        raise TypeError('pred and target should both be'
                        'torch.Tensor or np.ndarray')

    with torch.no_grad():
        res = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(0), min=1)
        res = res.mean().item() * 100
    return res


def f1_score(pred, target):
    """Calculate macro-averaged F1 score according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.

    Returns:
        float: The function will return a single float as F1 score.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        confusion_matrix = calculate_confusion_matrix_torch(pred, target)
    elif isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        confusion_matrix = calculate_confusion_matrix_numpy(pred, target)
    else:
        raise TypeError('pred and target should both be'
                        'torch.Tensor or np.ndarray')

    with torch.no_grad():
        precision = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(1), min=1)
        recall = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(0), min=1)
        res = 2 * precision * recall / torch.clamp(
            precision + recall, min=1e-20)
        res = torch.where(torch.isnan(res), torch.full_like(res, 0), res)
        res = res.mean().item() * 100
    return res

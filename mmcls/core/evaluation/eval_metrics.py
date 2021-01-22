import numpy as np
import torch


def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix with shape (C, C), where C is the number
             of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    num_classes = pred.size(1)
    _, pred_label = pred.topk(1, dim=1)
    pred_label = pred_label.view(-1)
    target_label = target.view(-1)
    assert len(pred_label) == len(target_label)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for t, p in zip(target_label, pred_label):
            confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def precision_recall_f1(pred, target, average_mode='macro', thr=None):
    """Calculate precision, recall and f1 score according to the prediction and
         target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.

        thr (float, optional): Predictions with scores under this threshold
            are considered negative. Default to None.

    Returns:
        float | np.array: Precision, recall, f1 score.
            The function returns a single float if the average_mode is set to
            macro, or a np.array with shape C if the average_mode is set to
             none.
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    label = np.indices(pred.shape)[1]
    pred_label = np.argsort(pred, axis=1)[:, -1]
    pred_score = np.sort(pred, axis=1)[:, -1]

    # Only prediction values larger than thr are counted as positive
    _pred_label = pred_label.copy()
    if thr is not None:
        _pred_label[pred_score <= thr] = -1
    pred_positive = label == _pred_label.reshape(-1, 1)
    gt_positive = label == target.reshape(-1, 1)
    precision = (pred_positive & gt_positive).sum(0) / np.maximum(
        pred_positive.sum(0), 1) * 100
    recall = (pred_positive & gt_positive).sum(0) / np.maximum(
        gt_positive.sum(0), 1) * 100
    f1_score = 2 * precision * recall / np.maximum(precision + recall, 1e-20)
    if average_mode == 'macro':
        precision = float(precision.mean())
        recall = float(recall.mean())
        f1_score = float(f1_score.mean())
    else:
        pass

    return precision, recall, f1_score


def precision(pred, target, average_mode='macro', thr=None):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thr (float, optional): Predictions with scores under this threshold
            are considered negative. Default to None.

    Returns:
        float | np.array: Precision, recall, f1 score.
            The function returns a single float if the average_mode is set to
            macro, or a np.array with shape C if the average_mode is set to
             none.
    """
    precision, _, _ = precision_recall_f1(pred, target, average_mode, thr)
    return precision


def recall(pred, target, average_mode='macro', thr=None):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thr (float, optional): Predictions with scores under this threshold
            are considered negative. Default to None.

    Returns:
        float | np.array: Precision, recall, f1 score.
            The function returns a single float if the average_mode is set to
            macro, or a np.array with shape C if the average_mode is set to
             none.
    """
    _, recall, _ = precision_recall_f1(pred, target, average_mode, thr)
    return recall


def f1_score(pred, target, average_mode='macro', thr=None):
    """Calculate F1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thr (float, optional): Predictions with scores under this threshold
            are considered negative. Default to None.

    Returns:
        float | np.array: Precision, recall, f1 score.
            The function returns a single float if the average_mode is set to
            macro, or a np.array with shape C if the average_mode is set to
             none.
    """
    _, _, f1_score = precision_recall_f1(pred, target, average_mode, thr)
    return f1_score


def support(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to
        the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: Precision, recall, f1 score.
            The function returns a single float if the average_mode is set to
            macro, or a np.array with shape C if the average_mode is set to
             none.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average_mode == 'macro':
            res = float(res.sum().numpy())
        elif average_mode == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res

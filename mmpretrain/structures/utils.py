# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn.functional as F
from mmengine.structures import LabelData

if hasattr(torch, 'tensor_split'):
    tensor_split = torch.tensor_split
else:
    # A simple implementation of `tensor_split`.
    def tensor_split(input: torch.Tensor, indices: list):
        outs = []
        for start, end in zip([0] + indices, indices + [input.size(0)]):
            outs.append(input[start:end])
        return outs


def cat_batch_labels(elements: List[LabelData], device=None):
    """Concat the ``label`` of a batch of :obj:`LabelData` to a tensor.

    Args:
        elements (List[LabelData]): A batch of :obj`LabelData`.
        device (torch.device, optional): The output device of the batch label.
            Defaults to None.

    Returns:
        Tuple[torch.Tensor, List[int]]: The first item is the concated label
        tensor, and the second item is the split indices of every sample.
    """
    item = elements[0]
    if 'label' not in item._data_fields:
        return None, None

    labels = []
    splits = [0]
    for element in elements:
        labels.append(element.label)
        splits.append(splits[-1] + element.label.size(0))
    batch_label = torch.cat(labels)
    if device is not None:
        batch_label = batch_label.to(device=device)
    return batch_label, splits[1:-1]


def batch_label_to_onehot(batch_label, split_indices, num_classes):
    """Convert a concated label tensor to onehot format.

    Args:
        batch_label (torch.Tensor): A concated label tensor from multiple
            samples.
        split_indices (List[int]): The split indices of every sample.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: The onehot format label tensor.

    Examples:
        >>> import torch
        >>> from mmpretrain.structures import batch_label_to_onehot
        >>> # Assume a concated label from 3 samples.
        >>> # label 1: [0, 1], label 2: [0, 2, 4], label 3: [3, 1]
        >>> batch_label = torch.tensor([0, 1, 0, 2, 4, 3, 1])
        >>> split_indices = [2, 5]
        >>> batch_label_to_onehot(batch_label, split_indices, num_classes=5)
        tensor([[1, 1, 0, 0, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]])
    """
    sparse_onehot_list = F.one_hot(batch_label, num_classes)
    onehot_list = [
        sparse_onehot.sum(0)
        for sparse_onehot in tensor_split(sparse_onehot_list, split_indices)
    ]
    return torch.stack(onehot_list)


def stack_batch_scores(elements, device=None):
    """Stack the ``score`` of a batch of :obj:`LabelData` to a tensor.

    Args:
        elements (List[LabelData]): A batch of :obj`LabelData`.
        device (torch.device, optional): The output device of the batch label.
            Defaults to None.

    Returns:
        torch.Tensor: The stacked score tensor.
    """
    item = elements[0]
    if 'score' not in item._data_fields:
        return None

    batch_score = torch.stack([element.score for element in elements])
    if device is not None:
        batch_score = batch_score.to(device)
    return batch_score

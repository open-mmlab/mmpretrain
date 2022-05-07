# Copyright (c) OpenMMLab. All rights reserved.

from numbers import Number
from typing import Sequence, Union

import mmcv
import numpy as np
import torch
from mmengine.data import BaseDataElement, LabelData


def format_label(value: Union[torch.Tensor, np.ndarray, Sequence, int],
                 num_classes: int = None) -> LabelData:
    """Convert label of various python types to :obj:`mmengine.LabelData`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.
        num_classes (int, optional): The number of classes. If not None, set
            it to the metainfo. Defaults to None.

    Returns:
        :obj:`mmengine.LabelData`: The foramtted label data.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmcv.is_str(value):
        value = torch.tensor(value)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')

    metainfo = {}
    if num_classes is not None:
        metainfo['num_classes'] = num_classes
        if value.max() >= num_classes:
            raise ValueError(f'The label data ({value}) should not '
                             f'exceed num_classes ({num_classes}).')
    label = LabelData(item=value, metainfo=metainfo)
    return label


class ClsDataSample(BaseDataElement):
    """A data structure interface of classification task.

    It's used as interfaces between different components.

    Meta field:
        img_shape (Tuple): The shape of the corresponding input image.
            Used for visualization.
        ori_shape (Tuple): The original shape of the corresponding image.
            Used for visualization.
        num_classes (int): The number of all categories.
            Used for label format conversion.

    Data field:
        gt_label (LabelData): The ground truth label.
        pred_label (LabelData): The predicted label.
        scores (torch.Tensor): The outputs of model.
        logits (torch.Tensor): The outputs of model without softmax nor
            sigmoid.

    Examples:
        >>> import torch
        >>> from mmcls.core import ClsDataSample
        >>>
        >>> img_meta = dict(img_shape=(960, 720), num_classes=5)
        >>> data_sample = ClsDataSample(metainfo=img_meta)
        >>> data_sample.set_gt_label(3)
        >>> print(data_sample)
        <ClsDataSample(
           META INFORMATION
           num_classes = 5
           img_shape = (960, 720)
           DATA FIELDS
           gt_label: <LabelData(
                   META INFORMATION
                   num_classes: 5
                   DATA FIELDS
                   label: tensor([3])
               ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
        >>> # For multi-label data
        >>> data_sample.set_gt_label([0, 1, 4])
        >>> print(data_sample.gt_label)
        <LabelData(
            META INFORMATION
            num_classes: 5
            DATA FIELDS
            item: tensor([0, 1, 4])
        ) at 0x7fd7d1b41970>
        >>> # Convert to one-hot format
        >>> data_sample.gt_label.to_onehot()
        >>> print(data_sample.gt_label)
        <LabelData(
            META INFORMATION
            num_classes: 5
            DATA FIELDS
            item: tensor([1, 1, 0, 0, 1])
        ) at 0x7fd7d1b41970>
    """

    def set_gt_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number],
                           Number]) -> None:
        """Set the gt_label data."""
        label = format_label(value, self.get('num_classes'))
        self.set_field(label, 'gt_label')

    def set_pred_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number],
                           Number]) -> None:
        """Set the pred_label data."""
        label = format_label(value, self.get('num_classes'))
        self.set_field(label, 'pred_label')

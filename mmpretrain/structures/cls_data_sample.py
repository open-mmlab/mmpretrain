# Copyright (c) OpenMMLab. All rights reserved.

from multiprocessing.reduction import ForkingPickler
from numbers import Number
from typing import Sequence, Union

import numpy as np
import torch
from mmengine.structures import BaseDataElement, LabelData
from mmengine.utils import is_str


def format_label(
        value: Union[torch.Tensor, np.ndarray, Sequence, int]) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The foramtted label tensor.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


def format_score(
        value: Union[torch.Tensor, np.ndarray, Sequence, int]) -> torch.Tensor:
    """Convert various python types to score-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence): Score values.

    Returns:
        :obj:`torch.Tensor`: The foramtted score tensor.
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


class ClsDataSample(BaseDataElement):
    """A data structure interface of classification task.

    It's used as interfaces between different components.

    Meta fields:
        img_shape (Tuple): The shape of the corresponding input image.
            Used for visualization.
        ori_shape (Tuple): The original shape of the corresponding image.
            Used for visualization.
        num_classes (int): The number of all categories.
            Used for label format conversion.

    Data fields:
        gt_label (:obj:`~mmengine.structures.LabelData`): The ground truth
            label.
        pred_label (:obj:`~mmengine.structures.LabelData`): The predicted
            label.
        scores (torch.Tensor): The outputs of model.
        logits (torch.Tensor): The outputs of model without softmax nor
            sigmoid.

    Examples:
        >>> import torch
        >>> from mmpretrain.structures import ClsDataSample
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
            label: tensor([0, 1, 4])
        ) at 0x7fd7d1b41970>
        >>> # Set one-hot format score
        >>> score = torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1])
        >>> data_sample.set_pred_score(score)
        >>> print(data_sample.pred_label)
        <LabelData(
            META INFORMATION
            num_classes: 5
            DATA FIELDS
            score: tensor([0.1, 0.1, 0.6, 0.1, 0.1])
        ) at 0x7fd7d1b41970>
    """

    def set_gt_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> 'ClsDataSample':
        """Set label of ``gt_label``."""
        label_data = getattr(self, '_gt_label', LabelData())
        label_data.label = format_label(value)
        self.gt_label = label_data
        return self

    def set_gt_score(self, value: torch.Tensor) -> 'ClsDataSample':
        """Set score of ``gt_label``."""
        label_data = getattr(self, '_gt_label', LabelData())
        label_data.score = format_score(value)
        if hasattr(self, 'num_classes'):
            assert len(label_data.score) == self.num_classes, \
                f'The length of score {len(label_data.score)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes',
                value=len(label_data.score),
                field_type='metainfo')
        self.gt_label = label_data
        return self

    def set_pred_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> 'ClsDataSample':
        """Set label of ``pred_label``."""
        label_data = getattr(self, '_pred_label', LabelData())
        label_data.label = format_label(value)
        self.pred_label = label_data
        return self

    def set_pred_score(self, value: torch.Tensor) -> 'ClsDataSample':
        """Set score of ``pred_label``."""
        label_data = getattr(self, '_pred_label', LabelData())
        label_data.score = format_score(value)
        if hasattr(self, 'num_classes'):
            assert len(label_data.score) == self.num_classes, \
                f'The length of score {len(label_data.score)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes',
                value=len(label_data.score),
                field_type='metainfo')
        self.pred_label = label_data
        return self

    @property
    def gt_label(self):
        return self._gt_label

    @gt_label.setter
    def gt_label(self, value: LabelData):
        self.set_field(value, '_gt_label', dtype=LabelData)

    @gt_label.deleter
    def gt_label(self):
        del self._gt_label

    @property
    def pred_label(self):
        return self._pred_label

    @pred_label.setter
    def pred_label(self, value: LabelData):
        self.set_field(value, '_pred_label', dtype=LabelData)

    @pred_label.deleter
    def pred_label(self):
        del self._pred_label


def _reduce_cls_datasample(data_sample):
    """reduce ClsDataSample."""
    attr_dict = data_sample.__dict__
    convert_keys = []
    for k, v in attr_dict.items():
        if isinstance(v, LabelData):
            attr_dict[k] = v.numpy()
            convert_keys.append(k)
    return _rebuild_cls_datasample, (attr_dict, convert_keys)


def _rebuild_cls_datasample(attr_dict, convert_keys):
    """rebuild ClsDataSample."""
    data_sample = ClsDataSample()
    for k in convert_keys:
        attr_dict[k] = attr_dict[k].to_tensor()
    data_sample.__dict__ = attr_dict
    return data_sample


# Due to the multi-processing strategy of PyTorch, ClsDataSample may consume
# many file descriptors because it contains multiple LabelData with tensors.
# Here we overwrite the reduce function of ClsDataSample in ForkingPickler and
# convert these tensors to np.ndarray during pickling. It may influence the
# performance of dataloader, but slightly because these tensors in LabelData
# are very small.
ForkingPickler.register(ClsDataSample, _reduce_cls_datasample)

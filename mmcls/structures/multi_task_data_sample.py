# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

from mmengine.structures import BaseDataElement, LabelData

from .cls_data_sample import ClsDataSample


def format_task_label(value: Dict, tasks: List[str]) -> LabelData:
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

    task_label = dict()
    for (key, val) in value.items():
        if key not in tasks:
            raise Exception(f'invalid task {key}.')
        task_label[key] = val
    label = LabelData(**task_label)
    return label


class MultiTaskDataSample(BaseDataElement):

    def __init__(self, tasks=None):
        super(MultiTaskDataSample, self).__init__()
        self.tasks = tasks

    def set_gt_label(self, value: Dict) -> 'MultiTaskDataSample':
        """Set label of ``gt_label``."""
        label = format_task_label(value, self.tasks)
        if 'gt_label' in self:
            self.gt_label.label = label.label
        else:
            self.gt_label = label
        return self

    def get_task_mask(self, task_name):
        return task_name in self.gt_label

    def get_task_sample(self, task_name):
        label = getattr(self.gt_label, task_name)
        label_task = ClsDataSample().set_gt_label(label)
        return label_task

    @property
    def gt_label(self):
        return self._gt_label

    @gt_label.setter
    def gt_label(self, value: LabelData):
        self.set_field(value, '_gt_label', dtype=LabelData)

    @gt_label.deleter
    def gt_label(self):
        del self._gt_label

# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict

import torch
from mmengine.structures import BaseDataElement, LabelData

from .cls_data_sample import ClsDataSample


def format_task_label(value: Dict, metainfo: Dict = {}) -> LabelData:
    """Convert label of various python types to :obj:`mmengine.LabelData`.

    Args:
        value : dict of  Label value.

    Returns:
        :obj:`mmengine.LabelData`: The foramtted label data.
    """

    # Handle single number

    task_label = dict()
    for (key, val) in value.items():
        if metainfo != {} and key not in metainfo.keys():
            raise Exception(f'Type {key} is not in metainfo.')
        task_label[key] = val
    label = LabelData(**task_label)
    return label


class MultiTaskDataSample(BaseDataElement):

    def set_gt_task(self, value: Dict) -> 'MultiTaskDataSample':
        """Set label of ``gt_task``."""
        label = format_task_label(value, self.metainfo)
        self.gt_task = label
        return self

    def set_pred_task(self, value: Dict) -> 'MultiTaskDataSample':
        """Set score of ``pred_task``."""
        self.pred_task = LabelData(**value)
        return self

    def get_task_mask(self, task_name):
        return task_name in self.gt_task

    @property
    def gt_task(self):
        return self._gt_task

    @gt_task.setter
    def gt_task(self, value: LabelData):
        self.set_field(value, '_gt_task', dtype=LabelData)

    @gt_task.deleter
    def gt_task(self):
        del self._gt_task

    @property
    def pred_task(self):
        return self._pred_task

    @pred_task.setter
    def pred_task(self, value: LabelData):
        self.set_field(value, '_pred_task', dtype=LabelData)

    @pred_task.deleter
    def pred_task(self):
        del self._pred_task

    def to_cls_data_sample(self, task_name):
        task_sample = ClsDataSample(metainfo=self.metainfo.get(task_name, {}))
        if hasattr(self, '_gt_task'):
            gt_task = getattr(self.gt_task, task_name, torch.tensor([]))
            task_sample.set_gt_label(value=gt_task)
        if hasattr(self, '_pred_task'):
            pred_task = getattr(self.pred_task, task_name, torch.tensor([]))
            task_sample.set_pred_score(value=pred_task)
        return task_sample

    def to_multi_task_data_sample(self, task_name):
        task_sample = MultiTaskDataSample(
            metainfo=self.metainfo.get(task_name, {}))
        if hasattr(self, '_gt_task'):
            gt_task = getattr(self.gt_task, task_name)
            task_sample.set_gt_task(value=gt_task)
        if hasattr(self, '_pred_task'):
            pred_task = getattr(self.pred_task, task_name)
            task_sample.set_pred_task(value=pred_task)
        return task_sample

    def to_target_data_sample(self, target_type, task_name):
        return self.data_samples_map[target_type](self, task_name)

    data_samples_map = {
        'ClsDataSample': to_cls_data_sample,
        'MultiTaskDataSample': to_multi_task_data_sample
    }

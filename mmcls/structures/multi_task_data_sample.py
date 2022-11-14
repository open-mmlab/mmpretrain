# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict

from mmengine.structures import BaseDataElement, LabelData

from .cls_data_sample import ClsDataSample


def format_task_label(value: Dict, metainfo: Dict = None) -> LabelData:
    """Convert label of various python types to :obj:`mmengine.LabelData`.

    Args:
        value : dict of  Label value.

    Returns:
        :obj:`mmengine.LabelData`: The foramtted label data.
    """

    # Handle single number

    task_label = dict()
    for (key, val) in value.items():
        if key not in metainfo.keys() and metainfo != {}:
            raise Exception(f'Type {key} is not in metainfo.')
        task_label[key] = val
    label = LabelData(**task_label)
    return label


class MultiTaskDataSample(BaseDataElement):

    def set_gt_task(self, value: Dict) -> 'MultiTaskDataSample':
        """Set label of ``gt_task``."""
        label = format_task_label(value, self.metainfo)
        if 'gt_task' in self:
            self.gt_task = label
        else:
            self.gt_task = label
        return self

    """
    def set_pred_task(self, value: Dict) -> 'MultiTaskDataSample':
        if 'pred_task' in self:
            self.pred_task = value
        else:
            self.pred_task = LabelData(**value)
        return self
    """
    def get_task_mask(self, task_name):
        return task_name in self.gt_task

    """
    def get_task_sample(self, task_name):
        label = self.gt_task[task_name]
        label_task = ClsDataSample().set_gt_task(label)
        return label_task
    """

    @property
    def gt_task(self):
        return self._gt_task

    @gt_task.setter
    def gt_task(self, value: LabelData):
        self.set_field(value, '_gt_task', dtype=LabelData)

    @gt_task.deleter
    def gt_task(self):
        del self._gt_task

    """
    @property
    def pred_task(self):
        return self._pred_task

    @pred_task.setter
    def pred_task(self, value: LabelData):
        self.set_field(value, '_pred_task', dtype=LabelData)

    @pred_task.deleter
    def pred_task(self):
        del self._pred_task
    """
    def to_cls_data_samples(self, task_name):
        label = getattr(self.gt_task, task_name)
        label_task = ClsDataSample(
            metainfo=self.metainfo.get(task_name, {})).set_gt_label(
                value=label)
        return label_task

    def to_multi_task_data_sample(self, task_name):
        label = getattr(self.gt_task, task_name)
        label_task = MultiTaskDataSample().set_gt_task(value=label)
        return label_task

    def to_target_data_sample(self, target_type, task_name):
        return self.data_samples_map[target_type](self, task_name)

    data_samples_map = {
        'ClsDataSample': to_cls_data_samples,
        'MultiTaskDataSample': to_multi_task_data_sample
    }

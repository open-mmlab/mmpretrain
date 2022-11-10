# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

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
    label = LabelData(label=task_label, metainfo=metainfo)
    return label


class MultiTaskDataSample(BaseDataElement):

    def set_gt_label(self, value: Dict) -> 'MultiTaskDataSample':
        """Set label of ``gt_label``."""
        label = format_task_label(value, self.metainfo)
        if 'gt_label' in self:
            self.gt_label.label = label.label
        else:
            self.gt_label = label
        return self

    def set_pred_score(self, value: Dict) -> 'MultiTaskDataSample':
        """Set label of ``pred_score``."""
        if 'pred_score' in self:
            self.pred_label.score = value
        else:
            self.pred_label = LabelData(score=value)
        return self

    def get_task_mask(self, task_name):
        return task_name in self.gt_label.label

    """
    def get_task_sample(self, task_name):
        label = self.gt_label.label[task_name]
        label_task = ClsDataSample().set_gt_label(label)
        return label_task
    """

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
    def pred_score(self):
        return self._pred_label

    @pred_score.setter
    def pred_score(self, value: LabelData):
        self.set_field(value, '_pred_label', dtype=LabelData)

    @pred_score.deleter
    def pred_score(self):
        del self._pred_label

    def to_cls_data_samples(self, task_name):
        label = self.gt_label.label[task_name]
        label_task = ClsDataSample(
            metainfo=self.metainfo.get(task_name, {})).set_gt_label(
                value=label)
        return label_task

    def to_multi_task_data_sample(self, task_name):
        label = self.gt_label.label[task_name]
        label_task = MultiTaskDataSample().set_gt_label(value=label)
        return label_task

    def to_target_data_sample(self, target_type, task_name):
        return self.data_samples_map[target_type](self, task_name)

    data_samples_map = {
        'ClsDataSample': to_cls_data_samples,
        'MultiTaskDataSample': to_multi_task_data_sample
    }

# Copyright (c) OpenMMLab. All rights reserved.
from .multi_label import AveragePrecision, MultiLabelMetric
from .multi_task import MultiTasks
from .multi_task_single_label import Accuracy_tasks, SingleLabelMetric_tasks
from .single_label import Accuracy, SingleLabelMetric

__all__ = [
    'Accuracy', 'SingleLabelMetric', 'MultiLabelMetric', 'AveragePrecision',
    'Accuracy_tasks', 'SingleLabelMetric_tasks', 'MultiTasks'
]

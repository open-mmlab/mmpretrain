# Copyright (c) OpenMMLab. All rights reserved.
from .multi_label import AveragePrecision, MultiLabelMetric
from .single_label import Accuracy, SingleLabelMetric
from .multi_task_single_label import Accuracy_tasks
__all__ = [
    'Accuracy', 'SingleLabelMetric', 'MultiLabelMetric', 'AveragePrecision','Accuracy_tasks'
]

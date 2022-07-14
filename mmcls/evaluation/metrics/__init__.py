# Copyright (c) OpenMMLab. All rights reserved.
from .multi_label import AveragePrecision, MultiLabelMetric
from .single_label import Accuracy, SingleLabelMetric

__all__ = [
    'Accuracy', 'SingleLabelMetric', 'MultiLabelMetric', 'AveragePrecision'
]

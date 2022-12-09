# Copyright (c) OpenMMLab. All rights reserved.
from .multi_label import AveragePrecision, MultiLabelMetric
from .single_label import Accuracy, SingleLabelMetric
from .voc_multi_label import VOCAveragePrecision, VOCMultiLabelMetric

__all__ = [
    'Accuracy', 'SingleLabelMetric', 'MultiLabelMetric', 'AveragePrecision',
    'VOCAveragePrecision', 'VOCMultiLabelMetric'
]

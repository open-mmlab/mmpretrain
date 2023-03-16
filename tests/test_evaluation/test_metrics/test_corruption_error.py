# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmcls.registry import METRICS


class TestCorruptionError(TestCase):

    def test_compute_metrics(self):
        mCE_metrics = METRICS.build(dict(type='CorruptionError'))
        results = [{
            'pred_score': torch.tensor([0.7, 0.0, 0.3]),
            'gt_label': torch.tensor([0]),
            'img_path': 'a/b/c/gaussian_noise'
        } for i in range(10)]
        metrics = mCE_metrics.compute_metrics(results)
        assert metrics['mCE'] == 0.0

    def test_process(self):
        mCE_metrics = METRICS.build(dict(type='CorruptionError'))
        results = [{
            'pred_label': {
                'label': torch.tensor([0]),
                'score': torch.tensor([0.7, 0.0, 0.3])
            },
            'gt_label': {
                'label': torch.tensor([0])
            },
            'img_path': 'a/b/c/gaussian_noise'
        } for i in range(10)]
        mCE_metrics.process(None, results)
        assert len(mCE_metrics.results) == 10

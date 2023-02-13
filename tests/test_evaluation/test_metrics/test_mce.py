# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
import torch
from mmcls.registry import METRICS


class TestAccuracy(TestCase):

    def test_compute_metrics(self):
        mCE_metrics = METRICS.build(dict(type='mCE'))
        results = [{
            'pred_score': torch.tensor([0.7, 0.0, 0.3]),
            'gt_label': torch.tensor([0]),
            'img_path': 'a/b/c/gaussian_noise'
        } for i in range(10)]
        metrics = mCE_metrics.compute_metrics(results)
        assert metrics['mCE'] == 0.0
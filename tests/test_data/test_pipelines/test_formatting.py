# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch

from mmcls.datasets import PIPELINES


class TestFormatMultiTaskLabels(TestCase):

    def test_call(self):
        results = {
            'task1_img_label': 1,
            'task2_img_label': [0, 1, 0, 1],
        }
        cfg = dict(type='FormatMultiTaskLabels', tasks=['task1', 'task2'])

        transform = PIPELINES.build(cfg)
        results_ = transform(deepcopy(results))
        self.assertIn('gt_label', results_)
        self.assertIsInstance(results_['gt_label']['task1'], torch.LongTensor)
        self.assertIsInstance(results_['gt_label']['task2'], torch.Tensor)

        # test auto parse task
        cfg = dict(type='FormatMultiTaskLabels')
        transform = PIPELINES.build(cfg)
        results_ = transform(deepcopy(results))
        self.assertIsInstance(results_['gt_label']['task1'], torch.LongTensor)
        self.assertIsInstance(results_['gt_label']['task2'], torch.Tensor)

    def test_repr(self):
        cfg = dict(type='FormatMultiTaskLabels', tasks=['task1', 'task2'])

        transform = PIPELINES.build(cfg)
        self.assertEquals(
            repr(transform), "FormatMultiTaskLabels(tasks=['task1', 'task2'])")

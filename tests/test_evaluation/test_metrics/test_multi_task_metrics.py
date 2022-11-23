# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmcls.evaluation.metrics import (MultiTasksMetric, Accuracy, SingleLabelMetric)
from mmcls.structures import MultiTaskDataSample


class MultiTaskMetric(TestCase):

    pred = [
        MultiTaskDataSample().set_pred_task(i).set_gt_task(
            k).to_dict() for i,  k in zip([
                {
                    'task0': torch.tensor([0.7, 0.0, 0.3]),
                    'task1': torch.tensor([0.5, 0.2, 0.3])
                },
                {
                    'task0': torch.tensor([0.0, 0.0, 1.0]),
                    'task1': torch.tensor([0.0, 0.0, 1.0])
                },
            ], [{'task0': 0, 'task1': 2}, {'task0': 2, 'task1': 1}])
    ]

    task_metrics = {
            'task0': [dict(type='Accuracy', topk=(1, ))],
            'task1': [
                dict(type='Accuracy', topk=(1, 3)),
                dict(type='SingleLabelMetric', items=['precision', 'recall'])
                ]
            }

    def test_init(self):
        metrics = MultiTasksMetric(self.task_metrics)
        self.assertIsInstance(metrics.Accuracy_task0, Accuracy)
        self.assertIsInstance(
            metrics.SingleLabelMetric_task1, SingleLabelMetric
        )

    def test_compute_metrics(self):
        results = [
            {
                'gt_task': {'task0': 0, 'task1': 2},
                'pred_task': {
                    'task0': torch.tensor([0.7, 0.0, 0.3]),
                    'task1': torch.tensor([0.5, 0.2, 0.3])
                },
            },
            {
                'gt_task': {'task0': 0, 'task1': 2},
                'pred_task': {
                    'task0': torch.tensor([0.0, 0.0, 1.0]),
                    'task1': torch.tensor([0.0, 0.0, 1.0])
                },
             }
        ]
        output = MultiTasksMetric(
            task_metrics=self.task_metrics).compute_metrics(results)
        self.assertIsInstance(output, dict)
        self.assertAlmostEqual(output['task0_accuracy/top1'], 1 / 2 * 100)
        self.assertGreater(output['task1_precision/top1'], 0)

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""

        # Test with score (use score instead of label if score exists)
        metric = MultiTasksMetric(self.task_metrics)
        metric.process(None, self.pred)
        results = metric.evaluate(2)
        self.assertIsInstance(results, dict)
        self.assertAlmostEqual(results['task0_accuracy/top1'], 2 / 6 * 100)
        self.assertGreater(results['task1_precision/top1'], 0)

# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmcls.evaluation.metrics import MultiTaskMetric
from mmcls.registry import METRICS
from mmcls.structures import MultiTaskDataSample


class MultiTasksMetric(TestCase):

    pred = [
        MultiTaskDataSample().set_pred_task(i).set_gt_task(
            k).to_dict() for i, j, k in zip([
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
                dict(type='precision', topk=(1, 3))]
            }

    def test_init_(self):
        metrics = MultiTasksMetric(self.task_metrics)

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
        output = MultiTaskMetric.compute_metrics(results)
        self.assertIsInstance(output, dict)
        self.assertAlmostEqual(output['task0_accuracy/top1'], 1 / 2 * 100)
        self.assertGreater(output['task0_accuracy/top1'], 0)

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""

        # Test with score (use score instead of label if score exists)
        metric = METRICS.build(dict(type='Accuracy', thrs=0.6))
        metric.process(None, self.pred)
        acc = metric.evaluate(2)
        self.assertIsInstance(acc, dict)
        self.assertAlmostEqual(acc['task0_accuracy/top1'], 2 / 6 * 100)

        # Test with multiple thrs
        metric = METRICS.build(dict(type='Accuracy', thrs=(0., 0.6, None)))
        metric.process(None, self.pred)
        acc = metric.evaluate(2)
        self.assertSetEqual(
            set(acc.keys()), {
                'accuracy/top1_thr-0.00', 'accuracy/top1_thr-0.60',
                'accuracy/top1_no-thr'
            })

        # Test with invalid topk
        with self.assertRaisesRegex(ValueError, 'check the `val_evaluator`'):
            metric = METRICS.build(dict(type='Accuracy', topk=(1, 5)))
            metric.process(None, self.pred)
            metric.evaluate(2)

        # Test initialization
        metric = METRICS.build(dict(type='Accuracy', thrs=0.6))
        self.assertTupleEqual(metric.thrs, (0.6, ))
        metric = METRICS.build(dict(type='Accuracy', thrs=[0.6]))
        self.assertTupleEqual(metric.thrs, (0.6, ))
        metric = METRICS.build(dict(type='Accuracy', topk=5))
        self.assertTupleEqual(metric.topk, (5, ))
        metric = METRICS.build(dict(type='Accuracy', topk=[5]))
        self.assertTupleEqual(metric.topk, (5, ))

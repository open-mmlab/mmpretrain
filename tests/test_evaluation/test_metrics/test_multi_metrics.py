# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import numpy as np
import torch

from mmcls.evaluation.metrics import Accuracy
from mmcls.registry import METRICS
from mmcls.structures import MultiTaskDataSample


class TestAccuracy(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
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

        # Test with score (use score instead of label if score exists)
        metric = METRICS.build(dict(type='Accuracy', thrs=0.6))
        metric.process(None, pred)
        acc = metric.evaluate(2)
        self.assertIsInstance(acc, dict)
        self.assertAlmostEqual(acc['accuracy/top1'], 2 / 6 * 100, places=4)

        # Test with multiple thrs
        metric = METRICS.build(dict(type='Accuracy', thrs=(0., 0.6, None)))
        metric.process(None, pred)
        acc = metric.evaluate(2)
        self.assertSetEqual(
            set(acc.keys()), {
                'accuracy/top1_thr-0.00', 'accuracy/top1_thr-0.60',
                'accuracy/top1_no-thr'
            })

        # Test with invalid topk
        with self.assertRaisesRegex(ValueError, 'check the `val_evaluator`'):
            metric = METRICS.build(dict(type='Accuracy', topk=(1, 5)))
            metric.process(None, pred)
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

    def test_calculate(self):
        """Test using the metric from static method."""

        # Test with score
        y_score = np.array({
            'task0': torch.tensor([0.7, 0.0, 0.3]),
            'task1': torch.tensor([0.5, 0.2, 0.3])
        },
        {
            'task0': torch.tensor([0.0, 0.0, 1.0]),
            'task1': torch.tensor([0.0, 0.0, 1.0]))
        y_true = [{'task0': 0, 'task1': 2}, {'task0': 2, 'task1': 1}]

        # Test with score
        acc = Accuracy.calculate(y_score, y_true, thrs=(0.6, ))
        self.assertIsInstance(acc, list)
        self.assertIsInstance(acc[0], list)
        self.assertIsInstance(acc[0][0], torch.Tensor)
        self.assertTensorEqual(acc[0][0], 2 / 6 * 100)

        # Test with label
        acc = Accuracy.calculate(y_label, y_true, thrs=(0.6, ))
        self.assertIsInstance(acc, torch.Tensor)
        # the thrs will be ignored
        self.assertTensorEqual(acc, 4 / 6 * 100)

        # Test with invalid inputs
        with self.assertRaisesRegex(TypeError, "<class 'str'> is not"):
            Accuracy.calculate(y_label, 'hi')

        # Test with invalid topk
        with self.assertRaisesRegex(ValueError, 'Top-5 accuracy .* is 3'):
            Accuracy.calculate(y_score, y_true, topk=(1, 5))

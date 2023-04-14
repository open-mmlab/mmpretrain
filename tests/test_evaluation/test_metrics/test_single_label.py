# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import numpy as np
import torch

from mmpretrain.evaluation.metrics import (Accuracy, ConfusionMatrix,
                                           SingleLabelMetric)
from mmpretrain.registry import METRICS
from mmpretrain.structures import DataSample


class TestAccuracy(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        pred = [
            DataSample().set_pred_score(i).set_pred_label(j).set_gt_label(
                k).to_dict() for i, j, k in zip([
                    torch.tensor([0.7, 0.0, 0.3]),
                    torch.tensor([0.5, 0.2, 0.3]),
                    torch.tensor([0.4, 0.5, 0.1]),
                    torch.tensor([0.0, 0.0, 1.0]),
                    torch.tensor([0.0, 0.0, 1.0]),
                    torch.tensor([0.0, 0.0, 1.0]),
                ], [0, 0, 1, 2, 2, 2], [0, 0, 1, 2, 1, 0])
        ]

        # Test with score (use score instead of label if score exists)
        metric = METRICS.build(dict(type='Accuracy', thrs=0.6))
        metric.process(None, pred)
        acc = metric.evaluate(6)
        self.assertIsInstance(acc, dict)
        self.assertAlmostEqual(acc['accuracy/top1'], 2 / 6 * 100, places=4)

        # Test with multiple thrs
        metric = METRICS.build(dict(type='Accuracy', thrs=(0., 0.6, None)))
        metric.process(None, pred)
        acc = metric.evaluate(6)
        self.assertSetEqual(
            set(acc.keys()), {
                'accuracy/top1_thr-0.00', 'accuracy/top1_thr-0.60',
                'accuracy/top1_no-thr'
            })

        # Test with invalid topk
        with self.assertRaisesRegex(ValueError, 'check the `val_evaluator`'):
            metric = METRICS.build(dict(type='Accuracy', topk=(1, 5)))
            metric.process(None, pred)
            metric.evaluate(6)

        # Test with label
        for sample in pred:
            del sample['pred_score']
        metric = METRICS.build(dict(type='Accuracy', thrs=(0., 0.6, None)))
        metric.process(None, pred)
        acc = metric.evaluate(6)
        self.assertIsInstance(acc, dict)
        self.assertAlmostEqual(acc['accuracy/top1'], 4 / 6 * 100, places=4)

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
        y_true = np.array([0, 0, 1, 2, 1, 0])
        y_label = torch.tensor([0, 0, 1, 2, 2, 2])
        y_score = [
            [0.7, 0.0, 0.3],
            [0.5, 0.2, 0.3],
            [0.4, 0.5, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]

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

    def assertTensorEqual(self,
                          tensor: torch.Tensor,
                          value: float,
                          msg=None,
                          **kwarg):
        tensor = tensor.to(torch.float32)
        value = torch.FloatTensor([value])
        try:
            torch.testing.assert_allclose(tensor, value, **kwarg)
        except AssertionError as e:
            self.fail(self._formatMessage(msg, str(e)))


class TestSingleLabel(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        pred = [
            DataSample().set_pred_score(i).set_pred_label(j).set_gt_label(
                k).to_dict() for i, j, k in zip([
                    torch.tensor([0.7, 0.0, 0.3]),
                    torch.tensor([0.5, 0.2, 0.3]),
                    torch.tensor([0.4, 0.5, 0.1]),
                    torch.tensor([0.0, 0.0, 1.0]),
                    torch.tensor([0.0, 0.0, 1.0]),
                    torch.tensor([0.0, 0.0, 1.0]),
                ], [0, 0, 1, 2, 2, 2], [0, 0, 1, 2, 1, 0])
        ]

        # Test with score (use score instead of label if score exists)
        metric = METRICS.build(
            dict(
                type='SingleLabelMetric',
                thrs=0.6,
                items=('precision', 'recall', 'f1-score', 'support')))
        metric.process(None, pred)
        res = metric.evaluate(6)
        self.assertIsInstance(res, dict)
        self.assertAlmostEqual(
            res['single-label/precision'], (1 + 0 + 1 / 3) / 3 * 100, places=4)
        self.assertAlmostEqual(
            res['single-label/recall'], (1 / 3 + 0 + 1) / 3 * 100, places=4)
        self.assertAlmostEqual(
            res['single-label/f1-score'], (1 / 2 + 0 + 1 / 2) / 3 * 100,
            places=4)
        self.assertEqual(res['single-label/support'], 6)

        # Test with multiple thrs
        metric = METRICS.build(
            dict(type='SingleLabelMetric', thrs=(0., 0.6, None)))
        metric.process(None, pred)
        res = metric.evaluate(6)
        self.assertSetEqual(
            set(res.keys()), {
                'single-label/precision_thr-0.00',
                'single-label/recall_thr-0.00',
                'single-label/f1-score_thr-0.00',
                'single-label/precision_thr-0.60',
                'single-label/recall_thr-0.60',
                'single-label/f1-score_thr-0.60',
                'single-label/precision_no-thr', 'single-label/recall_no-thr',
                'single-label/f1-score_no-thr'
            })

        # Test with average mode "micro"
        metric = METRICS.build(
            dict(
                type='SingleLabelMetric',
                average='micro',
                items=('precision', 'recall', 'f1-score', 'support')))
        metric.process(None, pred)
        res = metric.evaluate(6)
        self.assertIsInstance(res, dict)
        self.assertAlmostEqual(
            res['single-label/precision_micro'], 66.666, places=2)
        self.assertAlmostEqual(
            res['single-label/recall_micro'], 66.666, places=2)
        self.assertAlmostEqual(
            res['single-label/f1-score_micro'], 66.666, places=2)
        self.assertEqual(res['single-label/support_micro'], 6)

        # Test with average mode None
        metric = METRICS.build(
            dict(
                type='SingleLabelMetric',
                average=None,
                items=('precision', 'recall', 'f1-score', 'support')))
        metric.process(None, pred)
        res = metric.evaluate(6)
        self.assertIsInstance(res, dict)
        precision = res['single-label/precision_classwise']
        self.assertAlmostEqual(precision[0], 100., places=4)
        self.assertAlmostEqual(precision[1], 100., places=4)
        self.assertAlmostEqual(precision[2], 1 / 3 * 100, places=4)
        recall = res['single-label/recall_classwise']
        self.assertAlmostEqual(recall[0], 2 / 3 * 100, places=4)
        self.assertAlmostEqual(recall[1], 50., places=4)
        self.assertAlmostEqual(recall[2], 100., places=4)
        f1_score = res['single-label/f1-score_classwise']
        self.assertAlmostEqual(f1_score[0], 80., places=4)
        self.assertAlmostEqual(f1_score[1], 2 / 3 * 100, places=4)
        self.assertAlmostEqual(f1_score[2], 50., places=4)
        self.assertEqual(res['single-label/support_classwise'], [3, 2, 1])

        # Test with label, the thrs will be ignored
        pred_no_score = copy.deepcopy(pred)
        for sample in pred_no_score:
            del sample['pred_score']
            del sample['num_classes']
        metric = METRICS.build(
            dict(type='SingleLabelMetric', thrs=(0., 0.6), num_classes=3))
        metric.process(None, pred_no_score)
        res = metric.evaluate(6)
        self.assertIsInstance(res, dict)
        # Expected values come from sklearn
        self.assertAlmostEqual(res['single-label/precision'], 77.777, places=2)
        self.assertAlmostEqual(res['single-label/recall'], 72.222, places=2)
        self.assertAlmostEqual(res['single-label/f1-score'], 65.555, places=2)

        metric = METRICS.build(dict(type='SingleLabelMetric', thrs=(0., 0.6)))
        with self.assertRaisesRegex(AssertionError, 'must be specified'):
            metric.process(None, pred_no_score)

        # Test with empty items
        metric = METRICS.build(
            dict(type='SingleLabelMetric', items=tuple(), num_classes=3))
        metric.process(None, pred)
        res = metric.evaluate(6)
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 0)

        metric.process(None, pred_no_score)
        res = metric.evaluate(6)
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 0)

        # Test initialization
        metric = METRICS.build(dict(type='SingleLabelMetric', thrs=0.6))
        self.assertTupleEqual(metric.thrs, (0.6, ))
        metric = METRICS.build(dict(type='SingleLabelMetric', thrs=[0.6]))
        self.assertTupleEqual(metric.thrs, (0.6, ))

    def test_calculate(self):
        """Test using the metric from static method."""

        # Test with score
        y_true = np.array([0, 0, 1, 2, 1, 0])
        y_label = torch.tensor([0, 0, 1, 2, 2, 2])
        y_score = [
            [0.7, 0.0, 0.3],
            [0.5, 0.2, 0.3],
            [0.4, 0.5, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]

        # Test with score
        res = SingleLabelMetric.calculate(y_score, y_true, thrs=(0.6, ))
        self.assertIsInstance(res, list)
        self.assertIsInstance(res[0], tuple)
        precision, recall, f1_score, support = res[0]
        self.assertTensorEqual(precision, (1 + 0 + 1 / 3) / 3 * 100)
        self.assertTensorEqual(recall, (1 / 3 + 0 + 1) / 3 * 100)
        self.assertTensorEqual(f1_score, (1 / 2 + 0 + 1 / 2) / 3 * 100)
        self.assertTensorEqual(support, 6)

        # Test with label
        res = SingleLabelMetric.calculate(y_label, y_true, num_classes=3)
        self.assertIsInstance(res, tuple)
        precision, recall, f1_score, support = res
        # Expected values come from sklearn
        self.assertTensorEqual(precision, 77.7777)
        self.assertTensorEqual(recall, 72.2222)
        self.assertTensorEqual(f1_score, 65.5555)
        self.assertTensorEqual(support, 6)

        # Test with invalid inputs
        with self.assertRaisesRegex(TypeError, "<class 'str'> is not"):
            SingleLabelMetric.calculate(y_label, 'hi')

    def assertTensorEqual(self,
                          tensor: torch.Tensor,
                          value: float,
                          msg=None,
                          **kwarg):
        tensor = tensor.to(torch.float32)
        value = torch.tensor(value).float()
        try:
            torch.testing.assert_allclose(tensor, value, **kwarg)
        except AssertionError as e:
            self.fail(self._formatMessage(msg, str(e)))


class TestConfusionMatrix(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        pred = [
            DataSample().set_pred_score(i).set_pred_label(j).set_gt_label(
                k).to_dict() for i, j, k in zip([
                    torch.tensor([0.7, 0.0, 0.3]),
                    torch.tensor([0.5, 0.2, 0.3]),
                    torch.tensor([0.4, 0.5, 0.1]),
                    torch.tensor([0.0, 0.0, 1.0]),
                    torch.tensor([0.0, 0.0, 1.0]),
                    torch.tensor([0.0, 0.0, 1.0]),
                ], [0, 0, 1, 2, 2, 2], [0, 0, 1, 2, 1, 0])
        ]

        # Test with score (use score instead of label if score exists)
        metric = METRICS.build(dict(type='ConfusionMatrix'))
        metric.process(None, pred)
        res = metric.evaluate(6)
        self.assertIsInstance(res, dict)
        self.assertTensorEqual(
            res['confusion_matrix/result'],
            torch.tensor([
                [2, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
            ]))

        # Test with label
        for sample in pred:
            del sample['pred_score']
        metric = METRICS.build(dict(type='ConfusionMatrix'))
        metric.process(None, pred)
        with self.assertRaisesRegex(AssertionError,
                                    'Please specify the `num_classes`'):
            metric.evaluate(6)

        metric = METRICS.build(dict(type='ConfusionMatrix', num_classes=3))
        metric.process(None, pred)
        self.assertIsInstance(res, dict)
        self.assertTensorEqual(
            res['confusion_matrix/result'],
            torch.tensor([
                [2, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
            ]))

    def test_calculate(self):
        y_true = np.array([0, 0, 1, 2, 1, 0])
        y_label = torch.tensor([0, 0, 1, 2, 2, 2])
        y_score = [
            [0.7, 0.0, 0.3],
            [0.5, 0.2, 0.3],
            [0.4, 0.5, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]

        # Test with score
        cm = ConfusionMatrix.calculate(y_score, y_true)
        self.assertIsInstance(cm, torch.Tensor)
        self.assertTensorEqual(
            cm, torch.tensor([
                [2, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
            ]))

        # Test with label
        with self.assertRaisesRegex(AssertionError,
                                    'Please specify the `num_classes`'):
            ConfusionMatrix.calculate(y_label, y_true)

        cm = ConfusionMatrix.calculate(y_label, y_true, num_classes=3)
        self.assertIsInstance(cm, torch.Tensor)
        self.assertTensorEqual(
            cm, torch.tensor([
                [2, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
            ]))

        # Test with invalid inputs
        with self.assertRaisesRegex(TypeError, "<class 'str'> is not"):
            ConfusionMatrix.calculate(y_label, 'hi')

    def test_plot(self):
        import matplotlib.pyplot as plt

        cm = torch.tensor([[2, 0, 1], [0, 1, 1], [0, 0, 1]])
        fig = ConfusionMatrix.plot(cm, include_values=True, show=False)

        self.assertIsInstance(fig, plt.Figure)

    def assertTensorEqual(self,
                          tensor: torch.Tensor,
                          value: float,
                          msg=None,
                          **kwarg):
        tensor = tensor.to(torch.float32)
        value = torch.tensor(value).float()
        try:
            torch.testing.assert_allclose(tensor, value, **kwarg)
        except AssertionError as e:
            self.fail(self._formatMessage(msg, str(e)))

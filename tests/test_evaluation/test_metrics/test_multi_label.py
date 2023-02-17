# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import sklearn.metrics
import torch
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope

from mmpretrain.evaluation.metrics import AveragePrecision, MultiLabelMetric
from mmpretrain.structures import ClsDataSample

init_default_scope('mmpretrain')


class TestMultiLabel(TestCase):

    def test_calculate(self):
        """Test using the metric from static method."""

        y_true = [[0], [1, 3], [0, 1, 2], [3]]
        y_pred = [[0, 3], [0, 2], [1, 2], [2, 3]]
        y_true_binary = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
        ])
        y_pred_binary = np.array([
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ])
        y_pred_score = np.array([
            [0.8, 0, 0, 0.6],
            [0.2, 0, 0.6, 0],
            [0, 0.9, 0.6, 0],
            [0, 0, 0.2, 0.3],
        ])

        # Test with sequence of category indexes
        res = MultiLabelMetric.calculate(
            y_pred,
            y_true,
            pred_indices=True,
            target_indices=True,
            num_classes=4)
        self.assertIsInstance(res, tuple)
        precision, recall, f1_score, support = res
        expect_precision = sklearn.metrics.precision_score(
            y_true_binary, y_pred_binary, average='macro') * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true_binary, y_pred_binary, average='macro') * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true_binary, y_pred_binary, average='macro') * 100
        self.assertTensorEqual(precision, expect_precision)
        self.assertTensorEqual(recall, expect_recall)
        self.assertTensorEqual(f1_score, expect_f1)
        self.assertTensorEqual(support, 7)

        # Test with onehot input
        res = MultiLabelMetric.calculate(y_pred_binary,
                                         torch.from_numpy(y_true_binary))
        self.assertIsInstance(res, tuple)
        precision, recall, f1_score, support = res
        # Expected values come from sklearn
        self.assertTensorEqual(precision, expect_precision)
        self.assertTensorEqual(recall, expect_recall)
        self.assertTensorEqual(f1_score, expect_f1)
        self.assertTensorEqual(support, 7)

        # Test with topk argument
        res = MultiLabelMetric.calculate(
            y_pred_score, y_true, target_indices=True, topk=1, num_classes=4)
        self.assertIsInstance(res, tuple)
        precision, recall, f1_score, support = res
        # Expected values come from sklearn
        top1_y_pred = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        expect_precision = sklearn.metrics.precision_score(
            y_true_binary, top1_y_pred, average='macro') * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true_binary, top1_y_pred, average='macro') * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true_binary, top1_y_pred, average='macro') * 100
        self.assertTensorEqual(precision, expect_precision)
        self.assertTensorEqual(recall, expect_recall)
        self.assertTensorEqual(f1_score, expect_f1)
        self.assertTensorEqual(support, 7)

        # Test with thr argument
        res = MultiLabelMetric.calculate(
            y_pred_score, y_true, target_indices=True, thr=0.25, num_classes=4)
        self.assertIsInstance(res, tuple)
        precision, recall, f1_score, support = res
        # Expected values come from sklearn
        thr_y_pred = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ])
        expect_precision = sklearn.metrics.precision_score(
            y_true_binary, thr_y_pred, average='macro') * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true_binary, thr_y_pred, average='macro') * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true_binary, thr_y_pred, average='macro') * 100
        self.assertTensorEqual(precision, expect_precision)
        self.assertTensorEqual(recall, expect_recall)
        self.assertTensorEqual(f1_score, expect_f1)
        self.assertTensorEqual(support, 7)

        # Test with invalid inputs
        with self.assertRaisesRegex(TypeError, "<class 'str'> is not"):
            MultiLabelMetric.calculate(y_pred, 'hi', num_classes=10)

        # Test with invalid input
        with self.assertRaisesRegex(AssertionError,
                                    'Invalid `average` argument,'):
            MultiLabelMetric.calculate(
                y_pred, y_true, average='m', num_classes=10)

        y_true_binary = np.array([[1, 0, 0, 0], [0, 1, 0, 1]])
        y_pred_binary = np.array([[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]])
        # Test with invalid inputs
        with self.assertRaisesRegex(AssertionError, 'The size of pred'):
            MultiLabelMetric.calculate(y_pred_binary, y_true_binary)

        # Test with invalid inputs
        with self.assertRaisesRegex(TypeError, 'The `pred` and `target` must'):
            MultiLabelMetric.calculate(y_pred_binary, 5)

    def test_evaluate(self):
        y_true = [[0], [1, 3], [0, 1, 2], [3]]
        y_true_binary = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
        ])
        y_pred_score = torch.tensor([
            [0.8, 0, 0, 0.6],
            [0.2, 0, 0.6, 0],
            [0, 0.9, 0.6, 0],
            [0, 0, 0.2, 0.3],
        ])

        pred = [
            ClsDataSample(num_classes=4).set_pred_score(i).set_gt_label(j)
            for i, j in zip(y_pred_score, y_true)
        ]

        # Test with default argument
        evaluator = Evaluator(dict(type='MultiLabelMetric'))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)
        thr05_y_pred = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        expect_precision = sklearn.metrics.precision_score(
            y_true_binary, thr05_y_pred, average='macro') * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true_binary, thr05_y_pred, average='macro') * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true_binary, thr05_y_pred, average='macro') * 100
        self.assertEqual(res['multi-label/precision'], expect_precision)
        self.assertEqual(res['multi-label/recall'], expect_recall)
        self.assertEqual(res['multi-label/f1-score'], expect_f1)

        # Test with topk argument
        evaluator = Evaluator(dict(type='MultiLabelMetric', topk=1))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)
        top1_y_pred = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        expect_precision = sklearn.metrics.precision_score(
            y_true_binary, top1_y_pred, average='macro') * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true_binary, top1_y_pred, average='macro') * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true_binary, top1_y_pred, average='macro') * 100
        self.assertEqual(res['multi-label/precision_top1'], expect_precision)
        self.assertEqual(res['multi-label/recall_top1'], expect_recall)
        self.assertEqual(res['multi-label/f1-score_top1'], expect_f1)

        # Test with both argument
        evaluator = Evaluator(dict(type='MultiLabelMetric', thr=0.25, topk=1))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)
        # Expected values come from sklearn
        thr_y_pred = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ])
        expect_precision = sklearn.metrics.precision_score(
            y_true_binary, thr_y_pred, average='macro') * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true_binary, thr_y_pred, average='macro') * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true_binary, thr_y_pred, average='macro') * 100
        self.assertEqual(res['multi-label/precision_thr-0.25'],
                         expect_precision)
        self.assertEqual(res['multi-label/recall_thr-0.25'], expect_recall)
        self.assertEqual(res['multi-label/f1-score_thr-0.25'], expect_f1)

        # Test with average micro
        evaluator = Evaluator(dict(type='MultiLabelMetric', average='micro'))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)
        # Expected values come from sklearn
        expect_precision = sklearn.metrics.precision_score(
            y_true_binary, thr05_y_pred, average='micro') * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true_binary, thr05_y_pred, average='micro') * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true_binary, thr05_y_pred, average='micro') * 100
        self.assertAlmostEqual(
            res['multi-label/precision_micro'], expect_precision, places=4)
        self.assertAlmostEqual(
            res['multi-label/recall_micro'], expect_recall, places=4)
        self.assertAlmostEqual(
            res['multi-label/f1-score_micro'], expect_f1, places=4)

        # Test with average None
        evaluator = Evaluator(dict(type='MultiLabelMetric', average=None))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)
        # Expected values come from sklearn
        expect_precision = sklearn.metrics.precision_score(
            y_true_binary, thr05_y_pred, average=None) * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true_binary, thr05_y_pred, average=None) * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true_binary, thr05_y_pred, average=None) * 100
        np.testing.assert_allclose(res['multi-label/precision_classwise'],
                                   expect_precision)
        np.testing.assert_allclose(res['multi-label/recall_classwise'],
                                   expect_recall)
        np.testing.assert_allclose(res['multi-label/f1-score_classwise'],
                                   expect_f1)

        # Test with gt_score
        pred = [
            ClsDataSample(num_classes=4).set_pred_score(i).set_gt_score(j)
            for i, j in zip(y_pred_score, y_true_binary)
        ]

        evaluator = Evaluator(dict(type='MultiLabelMetric', items=['support']))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)
        self.assertEqual(res['multi-label/support'], 7)

    def assertTensorEqual(self,
                          tensor: torch.Tensor,
                          value: float,
                          msg=None,
                          **kwarg):
        tensor = tensor.to(torch.float32)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        value = torch.FloatTensor([value])
        try:
            torch.testing.assert_allclose(tensor, value, **kwarg)
        except AssertionError as e:
            self.fail(self._formatMessage(msg, str(e) + str(tensor)))


class TestAveragePrecision(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        y_pred = torch.tensor([
            [0.9, 0.8, 0.3, 0.2],
            [0.1, 0.2, 0.2, 0.1],
            [0.7, 0.5, 0.9, 0.3],
            [0.8, 0.1, 0.1, 0.2],
        ])
        y_true = torch.tensor([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ])

        pred = [
            ClsDataSample(num_classes=4).set_pred_score(i).set_gt_score(j)
            for i, j in zip(y_pred, y_true)
        ]

        # Test with default macro avergae
        evaluator = Evaluator(dict(type='AveragePrecision'))
        evaluator.process(pred)
        res = evaluator.evaluate(5)
        self.assertIsInstance(res, dict)
        self.assertAlmostEqual(res['multi-label/mAP'], 70.83333, places=4)

        # Test with average mode None
        evaluator = Evaluator(dict(type='AveragePrecision', average=None))
        evaluator.process(pred)
        res = evaluator.evaluate(5)
        self.assertIsInstance(res, dict)
        aps = res['multi-label/AP_classwise']
        self.assertAlmostEqual(aps[0], 100., places=4)
        self.assertAlmostEqual(aps[1], 83.3333, places=4)
        self.assertAlmostEqual(aps[2], 100, places=4)
        self.assertAlmostEqual(aps[3], 0, places=4)

        # Test with gt_label without score
        pred = [
            ClsDataSample(num_classes=4).set_pred_score(i).set_gt_label(j)
            for i, j in zip(y_pred, [[0, 1], [1], [2], [0]])
        ]
        evaluator = Evaluator(dict(type='AveragePrecision'))
        evaluator.process(pred)
        res = evaluator.evaluate(5)
        self.assertAlmostEqual(res['multi-label/mAP'], 70.83333, places=4)

    def test_calculate(self):
        """Test using the metric from static method."""

        y_true = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
        ])
        y_pred = np.array([
            [0.9, 0.8, 0.3, 0.2],
            [0.1, 0.2, 0.2, 0.1],
            [0.7, 0.5, 0.9, 0.3],
            [0.8, 0.1, 0.1, 0.2],
        ])

        ap_score = AveragePrecision.calculate(y_pred, y_true)
        expect_ap = sklearn.metrics.average_precision_score(y_true,
                                                            y_pred) * 100
        self.assertTensorEqual(ap_score, expect_ap)

        # Test with invalid inputs
        with self.assertRaisesRegex(AssertionError,
                                    'Invalid `average` argument,'):
            AveragePrecision.calculate(y_pred, y_true, average='m')

        y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 1]])
        y_pred = np.array([[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]])
        # Test with invalid inputs
        with self.assertRaisesRegex(AssertionError,
                                    'Both `pred` and `target`'):
            AveragePrecision.calculate(y_pred, y_true)

        # Test with invalid inputs
        with self.assertRaisesRegex(TypeError, "<class 'int'> is not an"):
            AveragePrecision.calculate(y_pred, 5)

    def assertTensorEqual(self,
                          tensor: torch.Tensor,
                          value: float,
                          msg=None,
                          **kwarg):
        tensor = tensor.to(torch.float32)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        value = torch.FloatTensor([value])
        try:
            torch.testing.assert_allclose(tensor, value, **kwarg)
        except AssertionError as e:
            self.fail(self._formatMessage(msg, str(e) + str(tensor)))

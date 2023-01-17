# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch

from mmcls.evaluation.metrics import RetrievalAveragePrecision, RetrievalRecall
from mmcls.registry import METRICS
from mmcls.structures import ClsDataSample


class TestRetrievalAveragePrecision(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        y_true = torch.tensor([[1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                               [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]])
        y_pred = torch.tensor([np.linspace(0.95, 0.05, 10)] * 2)

        pred = [
            ClsDataSample().set_pred_score(i).set_gt_score(j).to_dict()
            for i, j in zip(y_pred, y_true)
        ]

        # Test with default macro avergae
        evaluator = METRICS.build(
            dict(type='RetrievalAveragePrecision', topk=10))
        evaluator.process(None, pred)
        res = evaluator.evaluate(2)
        self.assertIsInstance(res, dict)
        self.assertAlmostEqual(
            res['retrieval/mAP@10'], 53.25396825396825, places=4)

        # Test with invalid topk
        with self.assertRaisesRegex(ValueError, '`topk` must be a'):
            evaluator = METRICS.build(
                dict(type='RetrievalAveragePrecision', topk=-1))

        # Test with invalid mode
        with self.assertRaisesRegex(AssertionError, 'Invalid `mode` '):
            evaluator = METRICS.build(
                dict(type='RetrievalAveragePrecision', topk=5, mode='m'))

    def test_calculate(self):
        """Test using the metric from static method."""
        # Test IR mode
        # example from https://zhuanlan.zhihu.com/p/35983818
        # or https://www.youtube.com/watch?v=pM6DJ0ZZee0

        # seq of indices format
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [np.arange(10)] * 2

        # test with average is 'macro'
        ap_score = RetrievalAveragePrecision.calculate(y_pred, y_true, True,
                                                       True)
        expect_ap = 53.25396728515625
        self.assertEqual(ap_score.item(), expect_ap)

        # test with tensor input
        y_true = torch.Tensor([[1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                               [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]])
        y_pred = np.array([np.linspace(0.95, 0.05, 10)] * 2)
        ap_score = RetrievalAveragePrecision.calculate(y_pred, y_true)
        expect_ap = 53.25396728515625
        self.assertEqual(ap_score.item(), expect_ap)

        # test with topk is 5
        y_pred = np.array([np.linspace(0.95, 0.05, 10)] * 2)
        ap_score = RetrievalAveragePrecision.calculate(y_pred, y_true, topk=5)
        expect_ap = 31.66666603088379
        self.assertEqual(ap_score.item(), expect_ap)

        # Test with invalid mode
        with self.assertRaisesRegex(AssertionError, 'Invalid `mode` '):
            RetrievalAveragePrecision.calculate(
                y_pred, y_true, True, True, mode='m')

        # Test with invalid pred
        y_pred = dict()
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        with self.assertRaisesRegex(AssertionError, '`pred` must be Seq'):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)

        # Test with invalid target
        y_true = dict()
        y_pred = [np.arange(10)] * 2
        with self.assertRaisesRegex(AssertionError, '`target` must be Seq'):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)

        # Test with different length `pred` with `target`
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [np.arange(10)] * 3
        with self.assertRaisesRegex(AssertionError, 'Length of `pred`'):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)

        # Test with invalid pred
        y_true = [[0, 2, 5, 8, 9], dict()]
        y_pred = [np.arange(10)] * 2
        with self.assertRaisesRegex(AssertionError, '`target` should be'):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)

        # Test with invalid target
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [np.arange(10), dict()]
        with self.assertRaisesRegex(AssertionError, '`pred` should be'):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)

        # Test with mode 'integrate'
        y_true = torch.Tensor([[1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                               [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]])
        y_pred = np.array([np.linspace(0.95, 0.05, 10)] * 2)

        ap_score = RetrievalAveragePrecision.calculate(
            y_pred, y_true, topk=5, mode='integrate')
        expect_ap = 25.41666603088379
        self.assertEqual(ap_score.item(), expect_ap)


class TestRetrievalRecall(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        pred = [
            ClsDataSample().set_pred_score(i).set_gt_label(k).to_dict()
            for i, k in zip([
                torch.tensor([0.7, 0.0, 0.3]),
                torch.tensor([0.5, 0.2, 0.3]),
                torch.tensor([0.4, 0.5, 0.1]),
                torch.tensor([0.0, 0.0, 1.0]),
                torch.tensor([0.0, 0.0, 1.0]),
                torch.tensor([0.0, 0.0, 1.0]),
            ], [[0], [0, 1], [1], [2], [1, 2], [0, 1]])
        ]

        # Test with score (use score instead of label if score exists)
        metric = METRICS.build(dict(type='RetrievalRecall', topk=1))
        metric.process(None, pred)
        recall = metric.evaluate(6)
        self.assertIsInstance(recall, dict)
        self.assertAlmostEqual(
            recall['retrieval/Recall@1'], 5 / 6 * 100, places=4)

        # Test with invalid topk
        with self.assertRaisesRegex(RuntimeError, 'selected index k'):
            metric = METRICS.build(dict(type='RetrievalRecall', topk=10))
            metric.process(None, pred)
            metric.evaluate(6)

        with self.assertRaisesRegex(ValueError, '`topk` must be a'):
            METRICS.build(dict(type='RetrievalRecall', topk=-1))

        # Test initialization
        metric = METRICS.build(dict(type='RetrievalRecall', topk=5))
        self.assertEqual(metric.topk, 5)

    def test_calculate(self):
        """Test using the metric from static method."""

        # seq of indices format
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [np.arange(10)] * 2

        # test with average is 'macro'
        ap_score = RetrievalRecall.calculate(
            y_pred, y_true, True, True, topk=1)
        expect_ap = 50.
        self.assertEqual(ap_score.item(), expect_ap)

        # test with tensor input
        y_true = torch.Tensor([[1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                               [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]])
        y_pred = np.array([np.linspace(0.95, 0.05, 10)] * 2)
        ap_score = RetrievalRecall.calculate(y_pred, y_true, topk=1)
        expect_ap = 50.
        self.assertEqual(ap_score.item(), expect_ap)

        # test with topk is 5
        y_pred = np.array([np.linspace(0.95, 0.05, 10)] * 2)
        ap_score = RetrievalRecall.calculate(y_pred, y_true, topk=5)
        expect_ap = 100.
        self.assertEqual(ap_score.item(), expect_ap)

        # Test with invalid pred
        y_pred = dict()
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        with self.assertRaisesRegex(AssertionError, '`pred` must be Seq'):
            RetrievalRecall.calculate(y_pred, y_true, True, True)

        # Test with invalid target
        y_true = dict()
        y_pred = [np.arange(10)] * 2
        with self.assertRaisesRegex(AssertionError, '`target` must be Seq'):
            RetrievalRecall.calculate(y_pred, y_true, True, True)

        # Test with different length `pred` with `target`
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [np.arange(10)] * 3
        with self.assertRaisesRegex(AssertionError, 'Length of `pred`'):
            RetrievalRecall.calculate(y_pred, y_true, True, True)

        # Test with invalid pred
        y_true = [[0, 2, 5, 8, 9], dict()]
        y_pred = [np.arange(10)] * 2
        with self.assertRaisesRegex(AssertionError, '`target` should be'):
            RetrievalRecall.calculate(y_pred, y_true, True, True)

        # Test with invalid target
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [np.arange(10), dict()]
        with self.assertRaisesRegex(AssertionError, '`pred` should be'):
            RetrievalRecall.calculate(y_pred, y_true, True, True)

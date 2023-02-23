# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import sklearn.metrics
import torch
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope

from mmpretrain.structures import DataSample

init_default_scope('mmpretrain')


class TestVOCMultiLabel(TestCase):

    def test_evaluate(self):
        # prepare input data
        y_true_label = [[0], [1, 3], [0, 1, 2], [3]]
        y_true_difficult = [[0], [2], [1], []]
        y_pred_score = torch.tensor([
            [0.8, 0, 0, 0.6],
            [0.2, 0, 0.6, 0],
            [0, 0.9, 0.6, 0],
            [0, 0, 0.2, 0.3],
        ])

        # generate data samples
        pred = [
            DataSample(num_classes=4).set_pred_score(i).set_gt_label(j)
            for i, j in zip(y_pred_score, y_true_label)
        ]
        for sample, difficult_label in zip(pred, y_true_difficult):
            sample.set_metainfo({'gt_label_difficult': difficult_label})

        # 1. Test with default argument
        evaluator = Evaluator(dict(type='VOCMultiLabelMetric'))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)

        # generate sklearn input
        y_true = np.array([
            [1, 0, 0, 0],
            [0, 1, -1, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
        ])
        ignored_index = y_true == -1
        y_true[ignored_index] = 0
        thr05_y_pred = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        thr05_y_pred[ignored_index] = 0

        expect_precision = sklearn.metrics.precision_score(
            y_true, thr05_y_pred, average='macro') * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true, thr05_y_pred, average='macro') * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true, thr05_y_pred, average='macro') * 100
        self.assertEqual(res['multi-label/precision'], expect_precision)
        self.assertEqual(res['multi-label/recall'], expect_recall)
        # precision is different between torch and sklearn
        self.assertAlmostEqual(res['multi-label/f1-score'], expect_f1, 5)

        # 2. Test with `difficult_as_positive`=False argument
        evaluator = Evaluator(
            dict(type='VOCMultiLabelMetric', difficult_as_positive=False))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)

        # generate sklearn input
        y_true = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
        ])
        thr05_y_pred = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])

        expect_precision = sklearn.metrics.precision_score(
            y_true, thr05_y_pred, average='macro') * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true, thr05_y_pred, average='macro') * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true, thr05_y_pred, average='macro') * 100
        self.assertEqual(res['multi-label/precision'], expect_precision)
        self.assertEqual(res['multi-label/recall'], expect_recall)
        # precision is different between torch and sklearn
        self.assertAlmostEqual(res['multi-label/f1-score'], expect_f1, 5)

        # 3. Test with `difficult_as_positive`=True argument
        evaluator = Evaluator(
            dict(type='VOCMultiLabelMetric', difficult_as_positive=True))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)

        # generate sklearn input
        y_true = np.array([
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
        ])
        thr05_y_pred = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])

        expect_precision = sklearn.metrics.precision_score(
            y_true, thr05_y_pred, average='macro') * 100
        expect_recall = sklearn.metrics.recall_score(
            y_true, thr05_y_pred, average='macro') * 100
        expect_f1 = sklearn.metrics.f1_score(
            y_true, thr05_y_pred, average='macro') * 100
        self.assertEqual(res['multi-label/precision'], expect_precision)
        self.assertEqual(res['multi-label/recall'], expect_recall)
        # precision is different between torch and sklearn
        self.assertAlmostEqual(res['multi-label/f1-score'], expect_f1, 5)


class TestVOCAveragePrecision(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        # prepare input data
        y_true_difficult = [[0], [2], [1], []]
        y_pred_score = torch.tensor([
            [0.8, 0.1, 0, 0.6],
            [0.2, 0.2, 0.7, 0],
            [0.1, 0.9, 0.6, 0.1],
            [0, 0, 0.2, 0.3],
        ])
        y_true_label = [[0], [1, 3], [0, 1, 2], [3]]
        y_true = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
        ])
        y_true_difficult = [[0], [2], [1], []]

        # generate data samples
        pred = [
            DataSample(num_classes=4).set_pred_score(i).set_gt_score(
                j).set_gt_label(k)
            for i, j, k in zip(y_pred_score, y_true, y_true_label)
        ]
        for sample, difficult_label in zip(pred, y_true_difficult):
            sample.set_metainfo({'gt_label_difficult': difficult_label})

        # 1. Test with default
        evaluator = Evaluator(dict(type='VOCAveragePrecision'))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)

        # prepare inputs for sklearn for this case
        y_pred_score = [[0.8, 0.2, 0.1, 0], [0.1, 0.2, 0.9, 0], [0, 0.6, 0.2],
                        [0.6, 0, 0.1, 0.3]]
        y_true = [[1, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0], [0, 1, 0, 1]]
        expected_res = []
        for pred_per_class, gt_per_class in zip(y_pred_score, y_true):
            expected_res.append(
                sklearn.metrics.average_precision_score(
                    gt_per_class, pred_per_class))

        self.assertAlmostEqual(
            res['multi-label/mAP'],
            sum(expected_res) * 100 / len(expected_res),
            places=4)

        # 2. Test with `difficult_as_positive`=False argument
        evaluator = Evaluator(
            dict(type='VOCAveragePrecision', difficult_as_positive=False))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)

        # prepare inputs for sklearn for this case
        y_pred_score = [[0.8, 0.2, 0.1, 0], [0.1, 0.2, 0.9, 0],
                        [0, 0.7, 0.6, 0.2], [0.6, 0, 0.1, 0.3]]
        y_true = [[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
        expected_res = []
        for pred_per_class, gt_per_class in zip(y_pred_score, y_true):
            expected_res.append(
                sklearn.metrics.average_precision_score(
                    gt_per_class, pred_per_class))

        self.assertAlmostEqual(
            res['multi-label/mAP'],
            sum(expected_res) * 100 / len(expected_res),
            places=4)

        # 3. Test with `difficult_as_positive`=True argument
        evaluator = Evaluator(
            dict(type='VOCAveragePrecision', difficult_as_positive=True))
        evaluator.process(pred)
        res = evaluator.evaluate(4)
        self.assertIsInstance(res, dict)

        # prepare inputs for sklearn for this case
        y_pred_score = [[0.8, 0.2, 0.1, 0], [0.1, 0.2, 0.9, 0],
                        [0, 0.7, 0.6, 0.2], [0.6, 0, 0.1, 0.3]]
        y_true = [[1, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1]]
        expected_res = []
        for pred_per_class, gt_per_class in zip(y_pred_score, y_true):
            expected_res.append(
                sklearn.metrics.average_precision_score(
                    gt_per_class, pred_per_class))

        self.assertAlmostEqual(
            res['multi-label/mAP'],
            sum(expected_res) * 100 / len(expected_res),
            places=4)

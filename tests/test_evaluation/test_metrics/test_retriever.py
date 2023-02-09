# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmcls.evaluation.metrics import RetrieverRecall
from mmcls.registry import METRICS
from mmcls.structures import ClsDataSample


class TestRetrieverRecall(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        pred = [
            ClsDataSample().set_pred_score(i).set_pred_label(j).set_gt_label(
                k).to_dict() for i, j, k in zip([
                    torch.tensor([0.7, 0.0, 0.3]),
                    torch.tensor([0.5, 0.2, 0.3]),
                    torch.tensor([0.4, 0.5, 0.1]),
                    torch.tensor([0.0, 0.0, 1.0]),
                    torch.tensor([0.0, 0.0, 1.0]),
                    torch.tensor([0.0, 0.0, 1.0]),
                ], [0, 0, 1, 2, 2, 2], [[0], [0, 1], [1], [2], [1, 2], [0, 1]])
        ]

        # Test with score (use score instead of label if score exists)
        metric = METRICS.build(dict(type='RetrieverRecall', thrs=0.6))
        metric.process(None, pred)
        recall = metric.evaluate(6)
        self.assertIsInstance(recall, dict)
        self.assertAlmostEqual(recall['recall/top1'], 3 / 6 * 100, places=4)

        # Test with multiple thrs
        metric = METRICS.build(
            dict(type='RetrieverRecall', thrs=(0., 0.6, None)))
        metric.process(None, pred)
        recall = metric.evaluate(6)
        self.assertSetEqual(
            set(recall.keys()), {
                'recall/top1_thr-0.00', 'recall/top1_thr-0.60',
                'recall/top1_no-thr'
            })

        # Test with invalid topk
        with self.assertRaisesRegex(ValueError, 'check the `val_evaluator`'):
            metric = METRICS.build(dict(type='RetrieverRecall', topk=(1, 5)))
            metric.process(None, pred)
            metric.evaluate(6)

        # Test with label
        for sample in pred:
            del sample['pred_label']['score']
        metric = METRICS.build(
            dict(type='RetrieverRecall', thrs=(0., 0.6, None)))
        metric.process(None, pred)
        recall = metric.evaluate(6)
        self.assertIsInstance(recall, dict)
        self.assertAlmostEqual(recall['recall/top1'], 5 / 6 * 100, places=4)

        # Test initialization
        metric = METRICS.build(dict(type='RetrieverRecall', thrs=0.6))
        self.assertTupleEqual(metric.thrs, (0.6, ))
        metric = METRICS.build(dict(type='RetrieverRecall', thrs=[0.6]))
        self.assertTupleEqual(metric.thrs, (0.6, ))
        metric = METRICS.build(dict(type='RetrieverRecall', topk=5))
        self.assertTupleEqual(metric.topk, (5, ))
        metric = METRICS.build(dict(type='RetrieverRecall', topk=[5]))
        self.assertTupleEqual(metric.topk, (5, ))

    def test_calculate(self):
        """Test using the metric from static method."""

        # Test with score
        y_true = [[0], [0, 1], [1], [2], [1, 2], [0, 1]]
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
        recall = RetrieverRecall.calculate(y_score, y_true, thrs=(0.6, ))
        self.assertIsInstance(recall, list)
        self.assertIsInstance(recall[0], list)
        self.assertIsInstance(recall[0][0], torch.Tensor)
        self.assertTensorEqual(recall[0][0], 3 / 6 * 100)

        # Test with label
        recall = RetrieverRecall.calculate(y_label, y_true, thrs=(0.6, ))
        self.assertIsInstance(recall, torch.Tensor)
        # the thrs will be ignored
        self.assertTensorEqual(recall, 5 / 6 * 100)

        # Test with invalid inputs
        with self.assertRaisesRegex(TypeError, "<class 'str'> is not"):
            RetrieverRecall.calculate(y_label, 'hi')

        # Test with invalid topk
        with self.assertRaisesRegex(ValueError, 'Top-5 recall .* is 3'):
            RetrieverRecall.calculate(y_score, y_true, topk=(1, 5))

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

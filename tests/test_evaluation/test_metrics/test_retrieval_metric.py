# Copyright (c) OpenMMLab. All rights reserved.
from pyexpat import model
from unittest import TestCase

import numpy as np

import torch
from mmengine.evaluator import Evaluator

from mmcls.evaluation.metrics.retrieval import RetrievalAveragePrecision
from mmcls.structures import ClsDataSample
from mmcls.utils import register_all_modules

register_all_modules()


class TestRetrievalAveragePrecision(TestCase):

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""
        y_true = torch.tensor([
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        ])
        y_pred = torch.tensor([ np.linspace(0.95,0.05, 10) ] * 2)

        pred = [
            ClsDataSample().set_pred_score(i).set_gt_score(j)
            for i, j in zip(y_pred, y_true)
        ]

        # Test with default macro avergae
        evaluator = Evaluator(dict(type='RetrievalAveragePrecision', topk=10))
        evaluator.process(pred)
        res = evaluator.evaluate(2)
        self.assertIsInstance(res, dict)
        self.assertAlmostEqual(res['retrieval/mAP@10'], 53.25396825396825, places=4)

        # Test with average mode None
        evaluator = Evaluator(dict(type='RetrievalAveragePrecision', topk=5, average=None))
        evaluator.process(pred)
        res = evaluator.evaluate(2)
        self.assertIsInstance(res, dict)
        aps = res['retrieval/AP_classwise@5']
        self.assertAlmostEqual(aps[0], 33.333333, places=4)
        self.assertAlmostEqual(aps[1], 30.000000, places=4)

        # Test with invalid topk
        with self.assertRaisesRegex(ValueError,'`topk` must be a'):
            evaluator = Evaluator(dict(type='RetrievalAveragePrecision', topk=-1))
        
        # Test with invalid average
        with self.assertRaisesRegex(AssertionError,'Invalid `average` '):
            evaluator = Evaluator(dict(type='RetrievalAveragePrecision', topk=5, average='m'))
        
        # Test with invalid mode
        with self.assertRaisesRegex(AssertionError,'Invalid `mode` '):
            evaluator = Evaluator(dict(type='RetrievalAveragePrecision', topk=5, mode='m'))



    def test_calculate(self):
        """Test using the metric from static method."""
        # Test IR mode
        # example from https://zhuanlan.zhihu.com/p/35983818 
        # or https://www.youtube.com/watch?v=pM6DJ0ZZee0
        
        # seq of indices format
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [ np.arange(10)] * 2

        # test with average is 'macro'
        ap_score = RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)
        expect_ap = 53.25396825396825
        self.assertEqual(ap_score.item(), expect_ap)

        # test with average is None
        ap_score = RetrievalAveragePrecision.calculate(y_pred, y_true, True, True, average=None)
        expect_ap = [62.22222222, 44.28571429]
        self.assertTrue(np.allclose(ap_score, np.array(expect_ap)))

        # test with tensor input
        y_true = torch.Tensor([[1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                               [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]])
        y_pred = np.array([ np.linspace(0.95,0.05, 10) ] * 2)
        ap_score = RetrievalAveragePrecision.calculate(y_pred, y_true)
        expect_ap = 53.25396825396825
        self.assertEqual(ap_score.item(), expect_ap)

        ap_score = RetrievalAveragePrecision.calculate(y_pred, y_true, average=None)
        expect_ap = [62.22222222, 44.28571429]
        self.assertTrue(np.allclose(ap_score, np.array(expect_ap)))

        # test with topk is 5
        y_pred = np.array([ np.linspace(0.95,0.05, 10) ] * 2)
        ap_score = RetrievalAveragePrecision.calculate(y_pred, y_true, topk=5)
        expect_ap = 31.666666666666664
        self.assertEqual(ap_score.item(), expect_ap)

        ap_score = RetrievalAveragePrecision.calculate(y_pred, y_true, topk=5, average=None)
        expect_ap = [33.333333, 30.0]
        self.assertTrue(np.allclose(ap_score, np.array(expect_ap)), msg=f"{ap_score}")


        # Test with invalid average
        with self.assertRaisesRegex(AssertionError,'Invalid `average` '):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True, average='m')
        
        # Test with invalid mode
        with self.assertRaisesRegex(AssertionError,'Invalid `mode` '):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True, mode='m')

        # Test with invalid pred
        y_pred = dict()
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        with self.assertRaisesRegex(AssertionError,'`pred` must be Seq'):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)
        
        # Test with invalid target
        y_true = dict()
        y_pred = [ np.arange(10)] * 2
        with self.assertRaisesRegex(AssertionError,'`target` must be Seq'):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)
        
        # Test with different length `pred` with `target`
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [ np.arange(10)] * 3
        with self.assertRaisesRegex(AssertionError,'Length of `pred`'):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)
        
        # Test with invalid pred
        y_true = [[0, 2, 5, 8, 9], dict()]
        y_pred = [ np.arange(10)] * 2
        with self.assertRaisesRegex(AssertionError,'`target` should be'):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)
        
        # Test with invalid target
        y_true = [[0, 2, 5, 8, 9], [1, 4, 6]]
        y_pred = [np.arange(10), dict()]
        with self.assertRaisesRegex(AssertionError,'`pred` should be'):
            RetrievalAveragePrecision.calculate(y_pred, y_true, True, True)

        # Test with mode 'integrate'
        y_true = torch.Tensor([[1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                               [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]])
        y_pred = np.array([ np.linspace(0.95,0.05, 10) ] * 2)

        ap_score = RetrievalAveragePrecision.calculate(
            y_pred, y_true, topk=5, mode="integrate")
        expect_ap = 25.416666666666664
        self.assertEqual(ap_score.item(), expect_ap)

        ap_score = RetrievalAveragePrecision.calculate(
            y_pred, y_true, topk=5, average=None, mode="integrate")
        expect_ap = [31.66666667, 19.16666667]
        self.assertTrue(np.allclose(ap_score, np.array(expect_ap)), msg=f"{ap_score}")

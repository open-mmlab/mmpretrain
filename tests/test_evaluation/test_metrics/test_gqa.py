# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.evaluator import Evaluator

from mmpretrain.structures import DataSample


class TestScienceQAMetric:

    def test_evaluate(self):
        meta_info = {
            'pred_answer': 'dog',
            'gt_answer': 'dog',
        }
        data_sample = DataSample(metainfo=meta_info)
        data_samples = [data_sample for _ in range(10)]
        evaluator = Evaluator(dict(type='mmpretrain.GQAAcc'))
        evaluator.process(data_samples)
        res = evaluator.evaluate(4)
        assert res['GQA/acc'] == 1.0

        meta_info = {
            'pred_answer': 'dog',
            'gt_answer': 'cat',
        }
        data_sample = DataSample(metainfo=meta_info)
        data_samples = [data_sample for _ in range(10)]
        evaluator = Evaluator(dict(type='mmpretrain.GQAAcc'))
        evaluator.process(data_samples)
        res = evaluator.evaluate(4)
        assert res['GQA/acc'] == 0.0

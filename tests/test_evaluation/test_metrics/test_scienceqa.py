# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.evaluator import Evaluator

from mmpretrain.structures import DataSample


class TestScienceQAMetric:

    def test_evaluate(self):
        meta_info = {
            'choices': ['A', 'B', 'C', 'D'],
            'pred_answer': 'A',
            'grade': 'grade1',
            'subject': 'language science',
            'gt_answer': 1,
            'hint': 'hint',
            'has_image': True
        }
        data_sample = DataSample(metainfo=meta_info)
        data_samples = [data_sample for _ in range(10)]
        evaluator = Evaluator(dict(type='mmpretrain.ScienceQAMetric'))
        evaluator.process(data_samples)
        res = evaluator.evaluate(4)
        assert res['acc_grade_1_6'] == 0.0
        assert res['acc_language'] == 0.0
        assert res['all_acc'] == 0.0

        meta_info = {
            'choices': ['A', 'B', 'C', 'D'],
            'pred_answer': 'A',
            'grade': 'grade1',
            'subject': 'language science',
            'gt_answer': 0,
            'hint': 'hint',
            'has_image': True
        }
        data_sample = DataSample(metainfo=meta_info)
        data_samples = [data_sample for _ in range(10)]
        evaluator = Evaluator(dict(type='mmpretrain.ScienceQAMetric'))
        evaluator.process(data_samples)
        res = evaluator.evaluate(4)
        assert res['acc_grade_1_6'] == 1.0
        assert res['acc_language'] == 1.0
        assert res['all_acc'] == 1.0

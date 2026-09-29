# Copyright (c) OpenMMLab. All rights reserved.
from mmpretrain.evaluation.metrics.ANLS import ANLS


class TestANLS:

    def test_trailing_spaces_do_not_shrink_the_distance(self):
        metric = ANLS()
        score = metric.compute_metrics([{
            'pred_answer': 'bbb   ',
            'gt_answer': ['a'],
        }])
        assert score['ANLS'] == 0

    def test_an_exact_answer_stays_a_full_score(self):
        metric = ANLS()
        score = metric.compute_metrics([{
            'pred_answer': 'yes ',
            'gt_answer': ['yes'],
        }])
        assert score['ANLS'] == 1

    def test_one_extra_letter_keeps_half(self):
        metric = ANLS()
        score = metric.compute_metrics([{
            'pred_answer': 'ab',
            'gt_answer': ['a'],
        }])
        assert score['ANLS'] == 0.5

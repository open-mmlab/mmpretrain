# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpretrain.evaluation.metrics import MultiTasksMetric
from mmpretrain.structures import ClsDataSample


class MultiTaskMetric(TestCase):
    data_pred = [
        {
            'task0': torch.tensor([0.7, 0.0, 0.3]),
            'task1': torch.tensor([0.5, 0.2, 0.3])
        },
        {
            'task0': torch.tensor([0.0, 0.0, 1.0]),
            'task1': torch.tensor([0.0, 0.0, 1.0])
        },
    ]
    data_gt = [{'task0': 0, 'task1': 2}, {'task1': 2}]

    preds = []
    for i, pred in enumerate(data_pred):
        sample = {}
        for task_name in pred:
            task_sample = ClsDataSample().set_pred_score(pred[task_name])
            if task_name in data_gt[i]:
                task_sample.set_gt_label(data_gt[i][task_name])
                task_sample.set_field(True, 'eval_mask', field_type='metainfo')
            else:
                task_sample.set_field(
                    False, 'eval_mask', field_type='metainfo')
            sample[task_name] = task_sample.to_dict()

        preds.append(sample)
    data2 = zip([
        {
            'task0': torch.tensor([0.7, 0.0, 0.3]),
            'task1': {
                'task10': torch.tensor([0.5, 0.2, 0.3]),
                'task11': torch.tensor([0.4, 0.3, 0.3])
            }
        },
        {
            'task0': torch.tensor([0.0, 0.0, 1.0]),
            'task1': {
                'task10': torch.tensor([0.1, 0.6, 0.3]),
                'task11': torch.tensor([0.5, 0.2, 0.3])
            }
        },
    ], [{
        'task0': 0,
        'task1': {
            'task10': 2,
            'task11': 0
        }
    }, {
        'task0': 2,
        'task1': {
            'task10': 1,
            'task11': 0
        }
    }])

    pred2 = []
    for score, label in data2:
        sample = {}
        for task_name in score:
            if type(score[task_name]) != dict:
                task_sample = ClsDataSample().set_pred_score(score[task_name])
                task_sample.set_gt_label(label[task_name])
                sample[task_name] = task_sample.to_dict()
                sample[task_name]['eval_mask'] = True
            else:
                sample[task_name] = {}
                sample[task_name]['eval_mask'] = True
                for task_name2 in score[task_name]:
                    task_sample = ClsDataSample().set_pred_score(
                        score[task_name][task_name2])
                    task_sample.set_gt_label(label[task_name][task_name2])
                    sample[task_name][task_name2] = task_sample.to_dict()
                    sample[task_name][task_name2]['eval_mask'] = True

        pred2.append(sample)

    pred3 = [{'task0': {'eval_mask': False}, 'task1': {'eval_mask': False}}]
    task_metrics = {
        'task0': [dict(type='Accuracy', topk=(1, ))],
        'task1': [
            dict(type='Accuracy', topk=(1, 3)),
            dict(type='SingleLabelMetric', items=['precision', 'recall'])
        ]
    }
    task_metrics2 = {
        'task0': [dict(type='Accuracy', topk=(1, ))],
        'task1': [
            dict(
                type='MultiTasksMetric',
                task_metrics={
                    'task10': [
                        dict(type='Accuracy', topk=(1, 3)),
                        dict(type='SingleLabelMetric', items=['precision'])
                    ],
                    'task11': [dict(type='Accuracy', topk=(1, ))]
                })
        ]
    }

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""

        # Test with score (use score instead of label if score exists)
        metric = MultiTasksMetric(self.task_metrics)
        metric.process(None, self.preds)
        results = metric.evaluate(2)
        self.assertIsInstance(results, dict)
        self.assertAlmostEqual(results['task0_accuracy/top1'], 100)
        self.assertGreater(results['task1_single-label/precision'], 0)

        # Test nested
        metric = MultiTasksMetric(self.task_metrics2)
        metric.process(None, self.pred2)
        results = metric.evaluate(2)
        self.assertIsInstance(results, dict)
        self.assertGreater(results['task1_task10_single-label/precision'], 0)
        self.assertGreater(results['task1_task11_accuracy/top1'], 0)

        # Test with without any ground truth value
        metric = MultiTasksMetric(self.task_metrics)
        metric.process(None, self.pred3)
        results = metric.evaluate(2)
        self.assertIsInstance(results, dict)
        self.assertEqual(results['task0_Accuracy'], 0)

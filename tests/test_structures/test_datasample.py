# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import LabelData

from mmcls.structures import ClsDataSample, MultiTaskDataSample


class TestClsDataSample(TestCase):

    def _test_set_label(self, key):
        data_sample = ClsDataSample()
        method = getattr(data_sample, 'set_' + key)
        # Test number
        method(1)
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test tensor with single number
        method(torch.tensor(2))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test array with single number
        method(np.array(3))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test tensor
        method(torch.tensor([1, 2, 3]))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.Tensor)
        self.assertTrue((label.label == torch.tensor([1, 2, 3])).all())

        # Test array
        method(np.array([1, 2, 3]))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertTrue((label.label == torch.tensor([1, 2, 3])).all())

        # Test Sequence
        method([1, 2, 3])
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertTrue((label.label == torch.tensor([1, 2, 3])).all())

        # Test unavailable type
        with self.assertRaisesRegex(TypeError, "<class 'str'> is not"):
            method('hi')

        # Test set num_classes
        data_sample = ClsDataSample(metainfo={'num_classes': 10})
        method = getattr(data_sample, 'set_' + key)
        method(5)
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIn('num_classes', label)
        self.assertEqual(label.num_classes, 10)

        # Test unavailable label
        with self.assertRaisesRegex(ValueError, r'data .*[15].* should '):
            method(15)

    def test_set_gt_label(self):
        self._test_set_label('gt_label')

    def test_set_pred_label(self):
        self._test_set_label('pred_label')

    def test_del_gt_label(self):
        data_sample = ClsDataSample()
        self.assertNotIn('gt_label', data_sample)
        data_sample.set_gt_label(1)
        self.assertIn('gt_label', data_sample)
        del data_sample.gt_label
        self.assertNotIn('gt_label', data_sample)

    def test_del_pred_label(self):
        data_sample = ClsDataSample()
        self.assertNotIn('pred_label', data_sample)
        data_sample.set_pred_label(1)
        self.assertIn('pred_label', data_sample)
        del data_sample.pred_label
        self.assertNotIn('pred_label', data_sample)

    def test_set_gt_score(self):
        data_sample = ClsDataSample(metainfo={'num_classes': 5})
        data_sample.set_gt_score(torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1]))
        self.assertIn('score', data_sample.gt_label)
        torch.testing.assert_allclose(data_sample.gt_label.score,
                                      [0.1, 0.1, 0.6, 0.1, 0.1])
        self.assertEqual(data_sample.gt_label.num_classes, 5)

        # Test set again
        data_sample.set_gt_score(torch.tensor([0.2, 0.1, 0.5, 0.1, 0.1]))
        torch.testing.assert_allclose(data_sample.gt_label.score,
                                      [0.2, 0.1, 0.5, 0.1, 0.1])

        # Test invalid type
        with self.assertRaisesRegex(AssertionError, 'be a torch.Tensor'):
            data_sample.set_gt_score([1, 2, 3])

        # Test invalid dims
        with self.assertRaisesRegex(AssertionError, 'but got 2'):
            data_sample.set_gt_score(torch.tensor([[0.1, 0.1, 0.6, 0.1, 0.1]]))

        # Test invalid num_classes
        with self.assertRaisesRegex(AssertionError, r'length of value \(4\)'):
            data_sample.set_gt_score(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        # Test auto inter num_classes
        data_sample = ClsDataSample()
        data_sample.set_gt_score(torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1]))
        self.assertEqual(data_sample.gt_label.num_classes, 5)

    def test_set_pred_score(self):
        data_sample = ClsDataSample(metainfo={'num_classes': 5})
        data_sample.set_pred_score(torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1]))
        self.assertIn('score', data_sample.pred_label)
        torch.testing.assert_allclose(data_sample.pred_label.score,
                                      [0.1, 0.1, 0.6, 0.1, 0.1])
        self.assertEqual(data_sample.pred_label.num_classes, 5)

        # Test set again
        data_sample.set_pred_score(torch.tensor([0.2, 0.1, 0.5, 0.1, 0.1]))
        torch.testing.assert_allclose(data_sample.pred_label.score,
                                      [0.2, 0.1, 0.5, 0.1, 0.1])

        # Test invalid type
        with self.assertRaisesRegex(AssertionError, 'be a torch.Tensor'):
            data_sample.set_pred_score([1, 2, 3])

        # Test invalid dims
        with self.assertRaisesRegex(AssertionError, 'but got 2'):
            data_sample.set_pred_score(
                torch.tensor([[0.1, 0.1, 0.6, 0.1, 0.1]]))

        # Test invalid num_classes
        with self.assertRaisesRegex(AssertionError, r'length of value \(4\)'):
            data_sample.set_pred_score(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        # Test auto inter num_classes
        data_sample = ClsDataSample()
        data_sample.set_pred_score(torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1]))
        self.assertEqual(data_sample.pred_label.num_classes, 5)


class TestMultiTaskDataSample(TestCase):

    def _test_set_label(self, key):
        data_sample = MultiTaskDataSample()
        method = getattr(data_sample, 'set_' + key)
        # Test Dict without metainfo
        method({'task0': 0, 'task1': 2})
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertEqual(label['task0'], 0)

        # Test empty Dict without metainfo
        method({})
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        with self.assertRaises(Exception):
            label['task0']

        data_sample2 = MultiTaskDataSample(metainfo={
            'task0': {
                'num_classes': 10
            },
            'task1': {
                'num_classes': 3
            }
        })
        method2 = getattr(data_sample2, 'set_' + key)

        # Test Dict with metainfo
        method2({'task0': 0, 'task1': 2})
        self.assertIn(key, data_sample2)
        label = getattr(data_sample2, key)
        self.assertIsInstance(label, LabelData)

        # Test empty Dict with metainfo
        method2({})
        self.assertIn(key, data_sample2)
        label = getattr(data_sample2, key)
        self.assertIsInstance(label, LabelData)
        with self.assertRaises(Exception):
            label['task0']

        # Test Dict with metainfo
        with self.assertRaises(Exception):
            method2({'task0': 0, 'task3': 2})

    def test_set_gt_label(self):
        self._test_set_label(key='gt_task')

    """
    def test_set_pred_score(self):
        data_sample = MultiTaskDataSample()
        data_sample.set_pred_score(torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1]))
        self.assertIn('score', data_sample.pred_label)
        torch.testing.assert_allclose(data_sample.pred_label.score,
                                      [0.1, 0.1, 0.6, 0.1, 0.1])

    """

    def test_get_task_mask(self):
        gt_label = {}
        gt_label['task0'] = 1
        data_sample = MultiTaskDataSample().set_gt_task(gt_label)
        self.assertTrue(data_sample.get_task_mask('task0'), True)
        self.assertFalse(data_sample.get_task_mask('task1'), True)

    def test_to_target_data_sample(self):
        gt_label = {}
        gt_label['task0'] = 1
        data_sample = MultiTaskDataSample().set_gt_task(gt_label)
        target_data_sample = data_sample.to_target_data_sample(
            'ClsDataSample', 'task0')
        self.assertIsInstance(target_data_sample, ClsDataSample)
        with self.assertRaises(Exception):
            data_sample.to_target_data_sample('ClsDataSample', 'task1')

        gt_label['task0'] = 'hi'
        data_sample = MultiTaskDataSample().set_gt_task(gt_label)
        with self.assertRaises(Exception):
            data_sample.to_target_data_sample('ClsDataSample', 'task0')

        gt_label['task0'] = 12
        data_sample = MultiTaskDataSample(metainfo={
            'task0': {
                'num_classes': 10
            }
        }).set_gt_task(gt_label)
        with self.assertRaises(Exception):
            data_sample.to_target_data_sample('ClsDataSample', 'task0')

        with self.assertRaises(Exception):
            data_sample.to_target_data_sample('MultiTaskDataSample', 'task0')

        gt_label = {'task0': {'task00': 0, 'task01': 1}}

        data_sample = MultiTaskDataSample().set_gt_task(gt_label)
        target_data_sample = data_sample.to_target_data_sample(
            'MultiTaskDataSample', 'task0')
        self.assertIsInstance(target_data_sample, MultiTaskDataSample)

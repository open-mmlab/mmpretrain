# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import LabelData

from mmpretrain.structures import ClsDataSample, MultiTaskDataSample


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
        data_sample = ClsDataSample()
        data_sample.set_gt_score(torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1]))
        self.assertIn('score', data_sample.gt_label)
        torch.testing.assert_allclose(data_sample.gt_label.score,
                                      [0.1, 0.1, 0.6, 0.1, 0.1])

        # Test set again
        data_sample.set_gt_score(torch.tensor([0.2, 0.1, 0.5, 0.1, 0.1]))
        torch.testing.assert_allclose(data_sample.gt_label.score,
                                      [0.2, 0.1, 0.5, 0.1, 0.1])

        # Test invalid length
        with self.assertRaisesRegex(AssertionError, 'should be equal to'):
            data_sample.set_gt_score([1, 2])

        # Test invalid dims
        with self.assertRaisesRegex(AssertionError, 'but got 2'):
            data_sample.set_gt_score(torch.tensor([[0.1, 0.1, 0.6, 0.1, 0.1]]))

    def test_set_pred_score(self):
        data_sample = ClsDataSample()
        data_sample.set_pred_score(torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1]))
        self.assertIn('score', data_sample.pred_label)
        torch.testing.assert_allclose(data_sample.pred_label.score,
                                      [0.1, 0.1, 0.6, 0.1, 0.1])

        # Test set again
        data_sample.set_pred_score(torch.tensor([0.2, 0.1, 0.5, 0.1, 0.1]))
        torch.testing.assert_allclose(data_sample.pred_label.score,
                                      [0.2, 0.1, 0.5, 0.1, 0.1])

        # Test invalid length
        with self.assertRaisesRegex(AssertionError, 'should be equal to'):
            data_sample.set_gt_score([1, 2])

        # Test invalid dims
        with self.assertRaisesRegex(AssertionError, 'but got 2'):
            data_sample.set_pred_score(
                torch.tensor([[0.1, 0.1, 0.6, 0.1, 0.1]]))


class TestMultiTaskDataSample(TestCase):

    def test_multi_task_data_sample(self):
        gt_label = {'task0': {'task00': 1, 'task01': 1}, 'task1': 1}
        data_sample = MultiTaskDataSample()
        task_sample = ClsDataSample().set_gt_label(gt_label['task1'])
        data_sample.set_field(task_sample, 'task1')
        data_sample.set_field(MultiTaskDataSample(), 'task0')
        for task_name in gt_label['task0']:
            task_sample = ClsDataSample().set_gt_label(
                gt_label['task0'][task_name])
            data_sample.task0.set_field(task_sample, task_name)
        self.assertIsInstance(data_sample.task0, MultiTaskDataSample)
        self.assertIsInstance(data_sample.task1, ClsDataSample)
        self.assertIsInstance(data_sample.task0.task00, ClsDataSample)

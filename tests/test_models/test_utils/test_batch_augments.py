# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from mmcls.core import ClsDataSample
from mmcls.models import Mixup, RandomBatchAugment
from mmcls.registry import BATCH_AUGMENTS

augment_cfgs = [
    dict(type='BatchCutMix', alpha=1., prob=1.),
    dict(type='BatchMixup', alpha=1., prob=1.),
    dict(type='Identity', prob=1.),
    dict(type='BatchResizeMix', alpha=1., prob=1.)
]


class TestRandomBatchAugment(TestCase):

    def test_initialize(self):
        # test single augmentation
        augments = dict(type='Mixup', alpha=1.)
        batch_augments = RandomBatchAugment(augments)
        self.assertIsInstance(batch_augments.augments, list)
        self.assertEqual(len(batch_augments.augments), 1)

        # test specify augments with object
        augments = Mixup(alpha=1.)
        batch_augments = RandomBatchAugment(augments)
        self.assertIsInstance(batch_augments.augments, list)
        self.assertEqual(len(batch_augments.augments), 1)

        # test multiple augmentation
        augments = [
            dict(type='Mixup', alpha=1.),
            dict(type='CutMix', alpha=0.8),
        ]
        batch_augments = RandomBatchAugment(augments)
        # mixup, cutmix
        self.assertEqual(len(batch_augments.augments), 2)
        self.assertIsNone(batch_augments.probs)

        # test specify probs
        augments = [
            dict(type='Mixup', alpha=1.),
            dict(type='CutMix', alpha=0.8),
        ]
        batch_augments = RandomBatchAugment(augments, probs=[0.5, 0.3])
        # mixup, cutmix and None
        self.assertEqual(len(batch_augments.augments), 3)
        self.assertAlmostEqual(batch_augments.probs[-1], 0.2)

        # test assertion
        with self.assertRaisesRegex(AssertionError, 'Got 2 vs 1'):
            RandomBatchAugment(augments, probs=0.5)

        with self.assertRaisesRegex(AssertionError, 'exceeds 1.'):
            RandomBatchAugment(augments, probs=[0.5, 0.6])

    def test_call(self):
        inputs = torch.rand(2, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1) for _ in range(2)]

        augments = [
            dict(type='Mixup', alpha=1.),
            dict(type='CutMix', alpha=0.8),
        ]
        batch_augments = RandomBatchAugment(augments, probs=[0.5, 0.3])

        with patch('numpy.random', np.random.RandomState(0)):
            batch_augments.augments[1] = MagicMock()
            batch_augments(inputs, data_samples)
            batch_augments.augments[1].assert_called_once_with(
                inputs, data_samples)

        augments = [
            dict(type='Mixup', alpha=1.),
            dict(type='CutMix', alpha=0.8),
        ]
        batch_augments = RandomBatchAugment(augments, probs=[0.0, 0.0])
        mixed_inputs, mixed_samples = batch_augments(inputs, data_samples)
        self.assertIs(mixed_inputs, inputs)
        self.assertIs(mixed_samples, data_samples)


class TestMixup(TestCase):
    DEFAULT_ARGS = dict(type='Mixup', alpha=1.)

    def test_initialize(self):
        with self.assertRaises(AssertionError):
            cfg = {**self.DEFAULT_ARGS, 'alpha': 'unknown'}
            BATCH_AUGMENTS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = {**self.DEFAULT_ARGS, 'num_classes': 'unknown'}
            BATCH_AUGMENTS.build(cfg)

    def test_call(self):
        inputs = torch.rand(2, 3, 224, 224)
        data_samples = [
            ClsDataSample(metainfo={
                'num_classes': 10
            }).set_gt_label(1) for _ in range(2)
        ]

        # test get num_classes from data_samples
        mixup = BATCH_AUGMENTS.build(self.DEFAULT_ARGS)
        mixed_inputs, mixed_samples = mixup(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (10, ))

        with self.assertRaisesRegex(RuntimeError, 'Not specify'):
            data_samples = [ClsDataSample().set_gt_label(1) for _ in range(2)]
            mixup(inputs, data_samples)

        # test binary classification
        cfg = {**self.DEFAULT_ARGS, 'num_classes': 1}
        mixup = BATCH_AUGMENTS.build(cfg)
        data_samples = [ClsDataSample().set_gt_label([]) for _ in range(2)]

        mixed_inputs, mixed_samples = mixup(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (1, ))

        # test multi-label classification
        cfg = {**self.DEFAULT_ARGS, 'num_classes': 5}
        mixup = BATCH_AUGMENTS.build(cfg)
        data_samples = [ClsDataSample().set_gt_label([1, 2]) for _ in range(2)]

        mixed_inputs, mixed_samples = mixup(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (5, ))


class TestCutMix(TestCase):
    DEFAULT_ARGS = dict(type='CutMix', alpha=1.)

    def test_initialize(self):
        with self.assertRaises(AssertionError):
            cfg = {**self.DEFAULT_ARGS, 'alpha': 'unknown'}
            BATCH_AUGMENTS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = {**self.DEFAULT_ARGS, 'num_classes': 'unknown'}
            BATCH_AUGMENTS.build(cfg)

    def test_call(self):
        inputs = torch.rand(2, 3, 224, 224)
        data_samples = [
            ClsDataSample(metainfo={
                'num_classes': 10
            }).set_gt_label(1) for _ in range(2)
        ]

        # test with cutmix_minmax
        cfg = {**self.DEFAULT_ARGS, 'cutmix_minmax': (0.1, 0.2)}
        cutmix = BATCH_AUGMENTS.build(cfg)
        mixed_inputs, mixed_samples = cutmix(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (10, ))

        # test without correct_lam
        cfg = {**self.DEFAULT_ARGS, 'correct_lam': False}
        cutmix = BATCH_AUGMENTS.build(cfg)
        mixed_inputs, mixed_samples = cutmix(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (10, ))

        # test get num_classes from data_samples
        cutmix = BATCH_AUGMENTS.build(self.DEFAULT_ARGS)
        mixed_inputs, mixed_samples = cutmix(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (10, ))

        with self.assertRaisesRegex(RuntimeError, 'Not specify'):
            data_samples = [ClsDataSample().set_gt_label(1) for _ in range(2)]
            cutmix(inputs, data_samples)

        # test binary classification
        cfg = {**self.DEFAULT_ARGS, 'num_classes': 1}
        cutmix = BATCH_AUGMENTS.build(cfg)
        data_samples = [ClsDataSample().set_gt_label([]) for _ in range(2)]

        mixed_inputs, mixed_samples = cutmix(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (1, ))

        # test multi-label classification
        cfg = {**self.DEFAULT_ARGS, 'num_classes': 5}
        cutmix = BATCH_AUGMENTS.build(cfg)
        data_samples = [ClsDataSample().set_gt_label([1, 2]) for _ in range(2)]

        mixed_inputs, mixed_samples = cutmix(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (5, ))


class TestResizeMix(TestCase):
    DEFAULT_ARGS = dict(type='ResizeMix', alpha=1.)

    def test_initialize(self):
        with self.assertRaises(AssertionError):
            cfg = {**self.DEFAULT_ARGS, 'alpha': 'unknown'}
            BATCH_AUGMENTS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = {**self.DEFAULT_ARGS, 'num_classes': 'unknown'}
            BATCH_AUGMENTS.build(cfg)

    def test_call(self):
        inputs = torch.rand(2, 3, 224, 224)
        data_samples = [
            ClsDataSample(metainfo={
                'num_classes': 10
            }).set_gt_label(1) for _ in range(2)
        ]

        # test get num_classes from data_samples
        mixup = BATCH_AUGMENTS.build(self.DEFAULT_ARGS)
        mixed_inputs, mixed_samples = mixup(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (10, ))

        with self.assertRaisesRegex(RuntimeError, 'Not specify'):
            data_samples = [ClsDataSample().set_gt_label(1) for _ in range(2)]
            mixup(inputs, data_samples)

        # test binary classification
        cfg = {**self.DEFAULT_ARGS, 'num_classes': 1}
        mixup = BATCH_AUGMENTS.build(cfg)
        data_samples = [ClsDataSample().set_gt_label([]) for _ in range(2)]

        mixed_inputs, mixed_samples = mixup(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (1, ))

        # test multi-label classification
        cfg = {**self.DEFAULT_ARGS, 'num_classes': 5}
        mixup = BATCH_AUGMENTS.build(cfg)
        data_samples = [ClsDataSample().set_gt_label([1, 2]) for _ in range(2)]

        mixed_inputs, mixed_samples = mixup(inputs, data_samples)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_samples[0].gt_label.score.shape, (5, ))

# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from mmpretrain.models import Mixup, RandomBatchAugment
from mmpretrain.registry import BATCH_AUGMENTS


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
        scores = torch.rand(2, 10)

        augments = [
            dict(type='Mixup', alpha=1.),
            dict(type='CutMix', alpha=0.8),
        ]
        batch_augments = RandomBatchAugment(augments, probs=[0.5, 0.3])

        with patch('numpy.random', np.random.RandomState(0)):
            batch_augments.augments[1] = MagicMock()
            batch_augments(inputs, scores)
            batch_augments.augments[1].assert_called_once_with(inputs, scores)

        augments = [
            dict(type='Mixup', alpha=1.),
            dict(type='CutMix', alpha=0.8),
        ]
        batch_augments = RandomBatchAugment(augments, probs=[0.0, 0.0])
        mixed_inputs, mixed_samples = batch_augments(inputs, scores)
        self.assertIs(mixed_inputs, inputs)
        self.assertIs(mixed_samples, scores)


class TestMixup(TestCase):
    DEFAULT_ARGS = dict(type='Mixup', alpha=1.)

    def test_initialize(self):
        with self.assertRaises(AssertionError):
            cfg = {**self.DEFAULT_ARGS, 'alpha': 'unknown'}
            BATCH_AUGMENTS.build(cfg)

    def test_call(self):
        inputs = torch.rand(2, 3, 224, 224)
        scores = torch.rand(2, 10)

        mixup = BATCH_AUGMENTS.build(self.DEFAULT_ARGS)
        mixed_inputs, mixed_scores = mixup(inputs, scores)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_scores.shape, (2, 10))

        # test binary classification
        scores = torch.rand(2, 1)

        mixed_inputs, mixed_scores = mixup(inputs, scores)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_scores.shape, (2, 1))


class TestCutMix(TestCase):
    DEFAULT_ARGS = dict(type='CutMix', alpha=1.)

    def test_initialize(self):
        with self.assertRaises(AssertionError):
            cfg = {**self.DEFAULT_ARGS, 'alpha': 'unknown'}
            BATCH_AUGMENTS.build(cfg)

    def test_call(self):
        inputs = torch.rand(2, 3, 224, 224)
        scores = torch.rand(2, 10)

        # test with cutmix_minmax
        cfg = {**self.DEFAULT_ARGS, 'cutmix_minmax': (0.1, 0.2)}
        cutmix = BATCH_AUGMENTS.build(cfg)
        mixed_inputs, mixed_scores = cutmix(inputs, scores)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_scores.shape, (2, 10))

        # test without correct_lam
        cfg = {**self.DEFAULT_ARGS, 'correct_lam': False}
        cutmix = BATCH_AUGMENTS.build(cfg)
        mixed_inputs, mixed_scores = cutmix(inputs, scores)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_scores.shape, (2, 10))

        # test default settings
        cutmix = BATCH_AUGMENTS.build(self.DEFAULT_ARGS)
        mixed_inputs, mixed_scores = cutmix(inputs, scores)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_scores.shape, (2, 10))

        # test binary classification
        scores = torch.rand(2, 1)

        mixed_inputs, mixed_scores = cutmix(inputs, scores)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_scores.shape, (2, 1))


class TestResizeMix(TestCase):
    DEFAULT_ARGS = dict(type='ResizeMix', alpha=1.)

    def test_initialize(self):
        with self.assertRaises(AssertionError):
            cfg = {**self.DEFAULT_ARGS, 'alpha': 'unknown'}
            BATCH_AUGMENTS.build(cfg)

    def test_call(self):
        inputs = torch.rand(2, 3, 224, 224)
        scores = torch.rand(2, 10)

        mixup = BATCH_AUGMENTS.build(self.DEFAULT_ARGS)
        mixed_inputs, mixed_scores = mixup(inputs, scores)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_scores.shape, (2, 10))

        # test binary classification
        scores = torch.rand(2, 1)

        mixed_inputs, mixed_scores = mixup(inputs, scores)
        self.assertEqual(mixed_inputs.shape, (2, 3, 224, 224))
        self.assertEqual(mixed_scores.shape, (2, 1))

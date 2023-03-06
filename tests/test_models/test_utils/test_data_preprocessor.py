# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmpretrain.models import (ClsDataPreprocessor, RandomBatchAugment,
                               SelfSupDataPreprocessor,
                               TwoNormDataPreprocessor, VideoDataPreprocessor)
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


class TestClsDataPreprocessor(TestCase):

    def test_stack_batch(self):
        cfg = dict(type='ClsDataPreprocessor')
        processor: ClsDataPreprocessor = MODELS.build(cfg)

        data = {
            'inputs': [torch.randint(0, 256, (3, 224, 224))],
            'data_samples': [DataSample().set_gt_label(1)]
        }
        processed_data = processor(data)
        inputs = processed_data['inputs']
        data_samples = processed_data['data_samples']
        self.assertEqual(inputs.shape, (1, 3, 224, 224))
        self.assertEqual(len(data_samples), 1)
        self.assertTrue((data_samples[0].gt_label == torch.tensor([1])).all())

    def test_padding(self):
        cfg = dict(type='ClsDataPreprocessor', pad_size_divisor=16)
        processor: ClsDataPreprocessor = MODELS.build(cfg)

        data = {
            'inputs': [
                torch.randint(0, 256, (3, 255, 255)),
                torch.randint(0, 256, (3, 224, 224))
            ]
        }
        inputs = processor(data)['inputs']
        self.assertEqual(inputs.shape, (2, 3, 256, 256))

        data = {'inputs': torch.randint(0, 256, (2, 3, 255, 255))}
        inputs = processor(data)['inputs']
        self.assertEqual(inputs.shape, (2, 3, 256, 256))

    def test_to_rgb(self):
        cfg = dict(type='ClsDataPreprocessor', to_rgb=True)
        processor: ClsDataPreprocessor = MODELS.build(cfg)

        data = {'inputs': [torch.randint(0, 256, (3, 224, 224))]}
        inputs = processor(data)['inputs']
        torch.testing.assert_allclose(data['inputs'][0].flip(0).float(),
                                      inputs[0])

        data = {'inputs': torch.randint(0, 256, (1, 3, 224, 224))}
        inputs = processor(data)['inputs']
        torch.testing.assert_allclose(data['inputs'].flip(1).float(), inputs)

    def test_normalization(self):
        cfg = dict(
            type='ClsDataPreprocessor',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5])
        processor: ClsDataPreprocessor = MODELS.build(cfg)

        data = {'inputs': [torch.randint(0, 256, (3, 224, 224))]}
        processed_data = processor(data)
        inputs = processed_data['inputs']
        self.assertTrue((inputs >= -1).all())
        self.assertTrue((inputs <= 1).all())
        self.assertIsNone(processed_data['data_samples'])

        data = {'inputs': torch.randint(0, 256, (1, 3, 224, 224))}
        inputs = processor(data)['inputs']
        self.assertTrue((inputs >= -1).all())
        self.assertTrue((inputs <= 1).all())

    def test_batch_augmentation(self):
        cfg = dict(
            type='ClsDataPreprocessor',
            num_classes=10,
            batch_augments=dict(augments=[
                dict(type='Mixup', alpha=0.8),
                dict(type='CutMix', alpha=1.)
            ]))
        processor: ClsDataPreprocessor = MODELS.build(cfg)
        self.assertIsInstance(processor.batch_augments, RandomBatchAugment)
        data = {
            'inputs': [torch.randint(0, 256, (3, 224, 224))],
            'data_samples': [DataSample().set_gt_label(1)]
        }
        processed_data = processor(data, training=True)
        self.assertIn('inputs', processed_data)
        self.assertIn('data_samples', processed_data)

        cfg['batch_augments'] = None
        processor: ClsDataPreprocessor = MODELS.build(cfg)
        self.assertIsNone(processor.batch_augments)
        data = {'inputs': [torch.randint(0, 256, (3, 224, 224))]}
        processed_data = processor(data, training=True)
        self.assertIn('inputs', processed_data)
        self.assertIsNone(processed_data['data_samples'])


class TestSelfSupDataPreprocessor(TestCase):

    def test_to_rgb(self):
        cfg = dict(type='SelfSupDataPreprocessor', to_rgb=True)
        processor: SelfSupDataPreprocessor = MODELS.build(cfg)
        self.assertTrue(processor._channel_conversion)

        fake_data = {
            'inputs':
            [torch.randn((2, 3, 224, 224)),
             torch.randn((2, 3, 224, 224))],
            'data_samples': [DataSample(), DataSample()]
        }
        inputs = processor(fake_data)['inputs']
        torch.testing.assert_allclose(fake_data['inputs'][0].flip(1).float(),
                                      inputs[0])
        torch.testing.assert_allclose(fake_data['inputs'][1].flip(1).float(),
                                      inputs[1])

    def test_forward(self):
        data_preprocessor = SelfSupDataPreprocessor(
            to_rgb=True, mean=[124, 117, 104], std=[59, 58, 58])

        # test list inputs
        fake_data = {
            'inputs': [torch.randn((2, 3, 224, 224))],
            'data_samples': [DataSample(), DataSample()]
        }
        fake_output = data_preprocessor(fake_data)
        self.assertEqual(len(fake_output['inputs']), 1)
        self.assertEqual(len(fake_output['data_samples']), 2)

        # test torch.Tensor inputs
        fake_data = {
            'inputs': torch.randn((2, 3, 224, 224)),
            'data_samples': [DataSample(), DataSample()]
        }
        fake_output = data_preprocessor(fake_data)
        self.assertEqual(fake_output['inputs'].shape,
                         torch.Size((2, 3, 224, 224)))
        self.assertEqual(len(fake_output['data_samples']), 2)


class TestTwoNormDataPreprocessor(TestCase):

    def test_assertion(self):
        with pytest.raises(AssertionError):
            _ = TwoNormDataPreprocessor(
                to_rgb=True,
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            )
        with pytest.raises(AssertionError):
            _ = TwoNormDataPreprocessor(
                to_rgb=True,
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
                second_mean=(127.5, 127.5),
                second_std=(127.5, 127.5, 127.5),
            )
        with pytest.raises(AssertionError):
            _ = TwoNormDataPreprocessor(
                to_rgb=True,
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
                second_mean=(127.5, 127.5, 127.5),
                second_std=(127.5, 127.5),
            )

    def test_forward(self):
        data_preprocessor = dict(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            second_mean=(127.5, 127.5, 127.5),
            second_std=(127.5, 127.5, 127.5),
            to_rgb=True)

        data_preprocessor = TwoNormDataPreprocessor(**data_preprocessor)
        fake_data = {
            'inputs':
            [torch.randn((2, 3, 224, 224)),
             torch.randn((2, 3, 224, 224))],
            'data_sample': [DataSample(), DataSample()]
        }
        fake_output = data_preprocessor(fake_data)
        self.assertEqual(len(fake_output['inputs']), 2)
        self.assertEqual(len(fake_output['data_samples']), 2)


class TestVideoDataPreprocessor(TestCase):

    def test_NCTHW_format(self):
        data_preprocessor = VideoDataPreprocessor(
            mean=[114.75, 114.75, 114.75],
            std=[57.375, 57.375, 57.375],
            to_rgb=True,
            format_shape='NCTHW')

        # test list inputs
        fake_data = {
            'inputs': [torch.randn((2, 3, 4, 224, 224))],
            'data_sample': [DataSample(), DataSample()]
        }
        fake_output = data_preprocessor(fake_data)
        self.assertEqual(len(fake_output['inputs']), 1)
        self.assertEqual(len(fake_output['data_samples']), 2)

        # test torch.Tensor inputs
        fake_data = {
            'inputs': torch.randn((2, 3, 4, 224, 224)),
            'data_sample': [DataSample(), DataSample()]
        }
        fake_output = data_preprocessor(fake_data)
        self.assertEqual(fake_output['inputs'].shape,
                         torch.Size((2, 3, 4, 224, 224)))
        self.assertEqual(len(fake_output['data_samples']), 2)

    def test_NCHW_format(self):
        data_preprocessor = VideoDataPreprocessor(
            mean=[114.75, 114.75, 114.75],
            std=[57.375, 57.375, 57.375],
            to_rgb=True,
            format_shape='NCHW')

        # test list inputs
        fake_data = {
            'inputs': [torch.randn((2, 3, 224, 224))],
            'data_sample': [DataSample(), DataSample()]
        }
        fake_output = data_preprocessor(fake_data)
        self.assertEqual(len(fake_output['inputs']), 1)
        self.assertEqual(len(fake_output['data_samples']), 2)

        # test torch.Tensor inputs
        fake_data = {
            'inputs': torch.randn((2, 3, 224, 224)),
            'data_sample': [DataSample(), DataSample()]
        }
        fake_output = data_preprocessor(fake_data)
        self.assertEqual(fake_output['inputs'].shape,
                         torch.Size((2, 3, 224, 224)))
        self.assertEqual(len(fake_output['data_samples']), 2)

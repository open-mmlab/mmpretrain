# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpretrain.models import ClsDataPreprocessor, RandomBatchAugment
from mmpretrain.registry import MODELS
from mmpretrain.structures import ClsDataSample


class TestClsDataPreprocessor(TestCase):

    def test_stack_batch(self):
        cfg = dict(type='ClsDataPreprocessor')
        processor: ClsDataPreprocessor = MODELS.build(cfg)

        data = {
            'inputs': [torch.randint(0, 256, (3, 224, 224))],
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }
        processed_data = processor(data)
        inputs = processed_data['inputs']
        data_samples = processed_data['data_samples']
        self.assertEqual(inputs.shape, (1, 3, 224, 224))
        self.assertEqual(len(data_samples), 1)
        self.assertTrue(
            (data_samples[0].gt_label.label == torch.tensor([1])).all())

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
            'data_samples': [ClsDataSample().set_gt_label(1)]
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

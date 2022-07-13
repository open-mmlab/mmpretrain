# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmcls.engine import ClsDataSample
from mmcls.models import ClsDataPreprocessor, RandomBatchAugment
from mmcls.registry import MODELS
from mmcls.utils import register_all_modules

register_all_modules()


class TestClsDataPreprocessor(TestCase):

    def test_stack_batch(self):
        cfg = dict(type='ClsDataPreprocessor')
        processor: ClsDataPreprocessor = MODELS.build(cfg)

        data = [{
            'inputs': torch.randint(0, 256, (3, 224, 224)),
            'data_sample': ClsDataSample().set_gt_label(1)
        }]
        inputs, data_samples = processor(data)
        self.assertEqual(inputs.shape, (1, 3, 224, 224))
        self.assertEqual(len(data_samples), 1)
        self.assertTrue(
            (data_samples[0].gt_label.label == torch.tensor([1])).all())

    def test_padding(self):
        cfg = dict(type='ClsDataPreprocessor', pad_size_divisor=16)
        processor: ClsDataPreprocessor = MODELS.build(cfg)

        data = [{
            'inputs': torch.randint(0, 256, (3, 255, 255))
        }, {
            'inputs': torch.randint(0, 256, (3, 224, 224))
        }]
        inputs, _ = processor(data)
        self.assertEqual(inputs.shape, (2, 3, 256, 256))

    def test_to_rgb(self):
        cfg = dict(type='ClsDataPreprocessor', to_rgb=True)
        processor: ClsDataPreprocessor = MODELS.build(cfg)

        data = [{'inputs': torch.randint(0, 256, (3, 224, 224))}]
        inputs, _ = processor(data)
        torch.testing.assert_allclose(
            data[0]['inputs'].flip(dims=(0, )).to(torch.float32), inputs[0])

    def test_normalization(self):
        cfg = dict(
            type='ClsDataPreprocessor',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5])
        processor: ClsDataPreprocessor = MODELS.build(cfg)

        data = [{'inputs': torch.randint(0, 256, (3, 224, 224))}]
        inputs, data_samples = processor(data)
        self.assertTrue((inputs >= -1).all())
        self.assertTrue((inputs <= 1).all())
        self.assertIsNone(data_samples)

    def test_batch_augmentation(self):
        cfg = dict(
            type='ClsDataPreprocessor',
            batch_augments=[
                dict(type='Mixup', alpha=0.8, num_classes=10),
                dict(type='CutMix', alpha=1., num_classes=10)
            ])
        processor: ClsDataPreprocessor = MODELS.build(cfg)
        self.assertIsInstance(processor.batch_augments, RandomBatchAugment)
        data = [{
            'inputs': torch.randint(0, 256, (3, 224, 224)),
            'data_sample': ClsDataSample().set_gt_label(1)
        }]
        _, data_samples = processor(data, training=True)

        cfg['batch_augments'] = None
        processor: ClsDataPreprocessor = MODELS.build(cfg)
        self.assertIsNone(processor.batch_augments)
        data = [{
            'inputs': torch.randint(0, 256, (3, 224, 224)),
        }]
        _, data_samples = processor(data, training=True)
        self.assertIsNone(data_samples)

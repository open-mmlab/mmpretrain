# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from mmengine import ConfigDict

from mmpretrain.models import ImageClassifier
from mmpretrain.registry import MODELS
from mmpretrain.structures import ClsDataSample


def has_timm() -> bool:
    try:
        import timm  # noqa: F401
        return True
    except ImportError:
        return False


def has_huggingface() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


class TestImageClassifier(TestCase):
    DEFAULT_ARGS = dict(
        type='ImageClassifier',
        backbone=dict(type='ResNet', depth=18),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss')))

    def test_initialize(self):
        model = MODELS.build(self.DEFAULT_ARGS)
        self.assertTrue(model.with_neck)
        self.assertTrue(model.with_head)

        cfg = {**self.DEFAULT_ARGS, 'pretrained': 'checkpoint'}
        model = MODELS.build(cfg)
        self.assertDictEqual(model.init_cfg,
                             dict(type='Pretrained', checkpoint='checkpoint'))

        cfg = ConfigDict(self.DEFAULT_ARGS)
        cfg.pop('neck')
        model = MODELS.build(cfg)
        self.assertFalse(model.with_neck)

        cfg = ConfigDict(self.DEFAULT_ARGS)
        cfg.pop('head')
        model = MODELS.build(cfg)
        self.assertFalse(model.with_head)

        # test set batch augmentation from train_cfg
        cfg = {
            **self.DEFAULT_ARGS, 'train_cfg':
            dict(augments=dict(type='Mixup', alpha=1.))
        }
        model: ImageClassifier = MODELS.build(cfg)
        self.assertIsNotNone(model.data_preprocessor.batch_augments)

        cfg = {**self.DEFAULT_ARGS, 'train_cfg': dict()}
        model: ImageClassifier = MODELS.build(cfg)
        self.assertIsNone(model.data_preprocessor.batch_augments)

    def test_extract_feat(self):
        inputs = torch.rand(1, 3, 224, 224)
        cfg = ConfigDict(self.DEFAULT_ARGS)
        cfg.backbone.out_indices = (0, 1, 2, 3)
        model: ImageClassifier = MODELS.build(cfg)

        # test backbone output
        feats = model.extract_feat(inputs, stage='backbone')
        self.assertEqual(len(feats), 4)
        self.assertEqual(feats[0].shape, (1, 64, 56, 56))
        self.assertEqual(feats[1].shape, (1, 128, 28, 28))
        self.assertEqual(feats[2].shape, (1, 256, 14, 14))
        self.assertEqual(feats[3].shape, (1, 512, 7, 7))

        # test neck output
        feats = model.extract_feat(inputs, stage='neck')
        self.assertEqual(len(feats), 4)
        self.assertEqual(feats[0].shape, (1, 64))
        self.assertEqual(feats[1].shape, (1, 128))
        self.assertEqual(feats[2].shape, (1, 256))
        self.assertEqual(feats[3].shape, (1, 512))

        # test pre_logits output
        feats = model.extract_feat(inputs, stage='pre_logits')
        self.assertEqual(feats.shape, (1, 512))

        # TODO: test transformer style feature extraction

        # test extract_feats
        multi_feats = model.extract_feats([inputs, inputs], stage='backbone')
        self.assertEqual(len(multi_feats), 2)
        for feats in multi_feats:
            self.assertEqual(len(feats), 4)
            self.assertEqual(feats[0].shape, (1, 64, 56, 56))
            self.assertEqual(feats[1].shape, (1, 128, 28, 28))
            self.assertEqual(feats[2].shape, (1, 256, 14, 14))
            self.assertEqual(feats[3].shape, (1, 512, 7, 7))

        # Without neck, return backbone
        cfg = ConfigDict(self.DEFAULT_ARGS)
        cfg.backbone.out_indices = (0, 1, 2, 3)
        cfg.pop('neck')
        model: ImageClassifier = MODELS.build(cfg)
        feats = model.extract_feat(inputs, stage='neck')
        self.assertEqual(len(feats), 4)
        self.assertEqual(feats[0].shape, (1, 64, 56, 56))
        self.assertEqual(feats[1].shape, (1, 128, 28, 28))
        self.assertEqual(feats[2].shape, (1, 256, 14, 14))
        self.assertEqual(feats[3].shape, (1, 512, 7, 7))

        # Without head, raise error
        cfg = ConfigDict(self.DEFAULT_ARGS)
        cfg.backbone.out_indices = (0, 1, 2, 3)
        cfg.pop('head')
        model: ImageClassifier = MODELS.build(cfg)
        with self.assertRaisesRegex(AssertionError, 'No head or the head'):
            model.extract_feat(inputs, stage='pre_logits')

        with self.assertRaisesRegex(AssertionError, 'use `extract_feat`'):
            model.extract_feats(inputs)

    def test_loss(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]

        model: ImageClassifier = MODELS.build(self.DEFAULT_ARGS)
        losses = model.loss(inputs, data_samples)
        self.assertGreater(losses['loss'].item(), 0)

    def test_predict(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]

        model: ImageClassifier = MODELS.build(self.DEFAULT_ARGS)
        predictions = model.predict(inputs)
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))

        predictions = model.predict(inputs, data_samples)
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))
        self.assertEqual(data_samples[0].pred_label.score.shape, (10, ))
        torch.testing.assert_allclose(data_samples[0].pred_label.score,
                                      predictions[0].pred_label.score)

    def test_forward(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]
        model: ImageClassifier = MODELS.build(self.DEFAULT_ARGS)

        # test pure forward
        outs = model(inputs)
        self.assertIsInstance(outs, torch.Tensor)

        # test forward train
        losses = model(inputs, data_samples, mode='loss')
        self.assertGreater(losses['loss'].item(), 0)

        # test forward test
        predictions = model(inputs, mode='predict')
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))

        predictions = model(inputs, data_samples, mode='predict')
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))
        self.assertEqual(data_samples[0].pred_label.score.shape, (10, ))
        torch.testing.assert_allclose(data_samples[0].pred_label.score,
                                      predictions[0].pred_label.score)

        # test forward with invalid mode
        with self.assertRaisesRegex(RuntimeError, 'Invalid mode "unknown"'):
            model(inputs, mode='unknown')

    def test_train_step(self):
        cfg = {
            **self.DEFAULT_ARGS, 'data_preprocessor':
            dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        }
        model: ImageClassifier = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 224, 224)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        optim_wrapper = MagicMock()
        log_vars = model.train_step(data, optim_wrapper)
        self.assertIn('loss', log_vars)
        optim_wrapper.update_params.assert_called_once()

    def test_val_step(self):
        cfg = {
            **self.DEFAULT_ARGS, 'data_preprocessor':
            dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        }
        model: ImageClassifier = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 224, 224)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        predictions = model.val_step(data)
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))

    def test_test_step(self):
        cfg = {
            **self.DEFAULT_ARGS, 'data_preprocessor':
            dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        }
        model: ImageClassifier = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 224, 224)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        predictions = model.test_step(data)
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))


@unittest.skipIf(not has_timm(), 'timm is not installed.')
class TestTimmClassifier(TestCase):
    DEFAULT_ARGS = dict(
        type='TimmClassifier',
        model_name='resnet18',
        loss=dict(type='CrossEntropyLoss'),
    )

    def test_initialize(self):
        model = MODELS.build(self.DEFAULT_ARGS)
        assert isinstance(model.model, nn.Module)

        # test set batch augmentation from train_cfg
        cfg = {
            **self.DEFAULT_ARGS, 'train_cfg':
            dict(augments=dict(type='Mixup', alpha=1.))
        }
        model: ImageClassifier = MODELS.build(cfg)
        self.assertIsNotNone(model.data_preprocessor.batch_augments)

        cfg = {**self.DEFAULT_ARGS, 'train_cfg': dict()}
        model: ImageClassifier = MODELS.build(cfg)
        self.assertIsNone(model.data_preprocessor.batch_augments)

    def test_loss(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]

        model: ImageClassifier = MODELS.build(self.DEFAULT_ARGS)
        losses = model.loss(inputs, data_samples)
        self.assertGreater(losses['loss'].item(), 0)

    def test_predict(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]

        model: ImageClassifier = MODELS.build(self.DEFAULT_ARGS)
        predictions = model.predict(inputs)
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))

        predictions = model.predict(inputs, data_samples)
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))
        self.assertEqual(data_samples[0].pred_label.score.shape, (1000, ))
        torch.testing.assert_allclose(data_samples[0].pred_label.score,
                                      predictions[0].pred_label.score)

    def test_forward(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]
        model: ImageClassifier = MODELS.build(self.DEFAULT_ARGS)

        # test pure forward
        outs = model(inputs)
        self.assertIsInstance(outs, torch.Tensor)

        # test forward train
        losses = model(inputs, data_samples, mode='loss')
        self.assertGreater(losses['loss'].item(), 0)

        # test forward test
        predictions = model(inputs, mode='predict')
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))

        predictions = model(inputs, data_samples, mode='predict')
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))
        self.assertEqual(data_samples[0].pred_label.score.shape, (1000, ))
        torch.testing.assert_allclose(data_samples[0].pred_label.score,
                                      predictions[0].pred_label.score)

        # test forward with invalid mode
        with self.assertRaisesRegex(RuntimeError, 'Invalid mode "unknown"'):
            model(inputs, mode='unknown')

    def test_train_step(self):
        cfg = {
            **self.DEFAULT_ARGS, 'data_preprocessor':
            dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        }
        model: ImageClassifier = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 224, 224)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        optim_wrapper = MagicMock()
        log_vars = model.train_step(data, optim_wrapper)
        self.assertIn('loss', log_vars)
        optim_wrapper.update_params.assert_called_once()

    def test_val_step(self):
        cfg = {
            **self.DEFAULT_ARGS, 'data_preprocessor':
            dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        }
        model: ImageClassifier = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 224, 224)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        predictions = model.val_step(data)
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))

    def test_test_step(self):
        cfg = {
            **self.DEFAULT_ARGS, 'data_preprocessor':
            dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        }
        model: ImageClassifier = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 224, 224)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        predictions = model.test_step(data)
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))


@unittest.skipIf(not has_huggingface(), 'huggingface is not installed.')
class TestHuggingFaceClassifier(TestCase):
    DEFAULT_ARGS = dict(
        type='HuggingFaceClassifier',
        model_name='microsoft/resnet-18',
        loss=dict(type='CrossEntropyLoss'),
    )

    def test_initialize(self):
        model = MODELS.build(self.DEFAULT_ARGS)
        assert isinstance(model.model, nn.Module)

        # test set batch augmentation from train_cfg
        cfg = {
            **self.DEFAULT_ARGS, 'train_cfg':
            dict(augments=dict(type='Mixup', alpha=1.))
        }
        model: ImageClassifier = MODELS.build(cfg)
        self.assertIsNotNone(model.data_preprocessor.batch_augments)

        cfg = {**self.DEFAULT_ARGS, 'train_cfg': dict()}
        model: ImageClassifier = MODELS.build(cfg)
        self.assertIsNone(model.data_preprocessor.batch_augments)

    def test_loss(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]

        model: ImageClassifier = MODELS.build(self.DEFAULT_ARGS)
        losses = model.loss(inputs, data_samples)
        self.assertGreater(losses['loss'].item(), 0)

    def test_predict(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]

        model: ImageClassifier = MODELS.build(self.DEFAULT_ARGS)
        predictions = model.predict(inputs)
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))

        predictions = model.predict(inputs, data_samples)
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))
        self.assertEqual(data_samples[0].pred_label.score.shape, (1000, ))
        torch.testing.assert_allclose(data_samples[0].pred_label.score,
                                      predictions[0].pred_label.score)

    def test_forward(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]
        model: ImageClassifier = MODELS.build(self.DEFAULT_ARGS)

        # test pure forward
        outs = model(inputs)
        self.assertIsInstance(outs, torch.Tensor)

        # test forward train
        losses = model(inputs, data_samples, mode='loss')
        self.assertGreater(losses['loss'].item(), 0)

        # test forward test
        predictions = model(inputs, mode='predict')
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))

        predictions = model(inputs, data_samples, mode='predict')
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))
        self.assertEqual(data_samples[0].pred_label.score.shape, (1000, ))
        torch.testing.assert_allclose(data_samples[0].pred_label.score,
                                      predictions[0].pred_label.score)

        # test forward with invalid mode
        with self.assertRaisesRegex(RuntimeError, 'Invalid mode "unknown"'):
            model(inputs, mode='unknown')

    def test_train_step(self):
        cfg = {
            **self.DEFAULT_ARGS, 'data_preprocessor':
            dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        }
        model: ImageClassifier = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 224, 224)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        optim_wrapper = MagicMock()
        log_vars = model.train_step(data, optim_wrapper)
        self.assertIn('loss', log_vars)
        optim_wrapper.update_params.assert_called_once()

    def test_val_step(self):
        cfg = {
            **self.DEFAULT_ARGS, 'data_preprocessor':
            dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        }
        model: ImageClassifier = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 224, 224)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        predictions = model.val_step(data)
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))

    def test_test_step(self):
        cfg = {
            **self.DEFAULT_ARGS, 'data_preprocessor':
            dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        }
        model: ImageClassifier = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 224, 224)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        predictions = model.test_step(data)
        self.assertEqual(predictions[0].pred_label.score.shape, (1000, ))

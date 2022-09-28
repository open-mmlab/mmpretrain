# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine import ConfigDict

from mmcls.models import ImageClassifier
from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from mmcls.utils import register_all_modules

register_all_modules()


class TestImageToImageRetriever(TestCase):
    DEFAULT_ARGS = dict(
        type='ImageToImageRetriever',
        prototype_encoder=[
            dict(type='ResNet', depth=18, out_indices=(3, )),
            dict(type='GlobalAveragePooling'),
        ],
        head=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss')),
        prototype=torch.randn((10, 512)),
        dim=512)

    def test_initialize(self):
        model = MODELS.build(self.DEFAULT_ARGS)
        self.assertEqual(model.dim, 512)
        # test proto_pype
        self.assertEqual(type(model.prototype), torch.Tensor)
        self.assertFalse(model.prototype_inited)
        self.assertIsNone(model.prototype_vecs)

        cfg = {**self.DEFAULT_ARGS, 'pretrained': 'checkpoint'}
        model = MODELS.build(cfg)
        self.assertDictEqual(model.init_cfg,
                             dict(type='Pretrained', checkpoint='checkpoint'))

        # test set batch augmentation from train_cfg
        cfg = {
            **self.DEFAULT_ARGS, 'train_cfg':
            dict(augments=dict(type='Mixup', alpha=1., num_classes=10))
        }
        model = MODELS.build(cfg)
        self.assertIsNotNone(model.data_preprocessor.batch_augments)

        cfg = {**self.DEFAULT_ARGS, 'train_cfg': dict()}
        model = MODELS.build(cfg)
        self.assertIsNone(model.data_preprocessor.batch_augments)

    def test_extract_feat(self):
        inputs = torch.rand(1, 3, 224, 224)
        cfg = ConfigDict(self.DEFAULT_ARGS)
        model = MODELS.build(cfg)

        # test backbone output
        feats = model.extract_feat(inputs)
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].shape, (1, 512))

    def test_loss(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]

        model: ImageClassifier = MODELS.build(self.DEFAULT_ARGS)
        losses = model.loss(inputs, data_samples)
        self.assertGreater(losses['loss'].item(), 0)

    def test_prepare_prototype(self):
        # tensor
        cfg = {**self.DEFAULT_ARGS, 'prototype': torch.randn((10, 512))}
        model = MODELS.build(cfg)
        model.prepare_prototype()
        self.assertEqual(type(model.prototype_vecs), torch.Tensor)
        self.assertEqual(model.prototype_vecs.shape, (10, 512))
        self.assertTrue(model.prototype_inited)

    def test_predict(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label([1, 2, 6])]
        # default
        model = MODELS.build(self.DEFAULT_ARGS)
        predictions = model.predict(inputs)
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))

        predictions = model.predict(inputs, data_samples)
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))
        self.assertEqual(data_samples[0].pred_label.score.shape, (10, ))
        torch.testing.assert_allclose(data_samples[0].pred_label.score,
                                      predictions[0].pred_label.score)

        # k is not -1
        cfg = {**self.DEFAULT_ARGS, 'topk': 2}
        model = MODELS.build(cfg)

        predictions = model.predict(inputs)
        self.assertEqual(predictions[0].pred_label.score.shape, (2, ))

        predictions = model.predict(inputs, data_samples)
        self.assertEqual(predictions[0].pred_label.score.shape, (2, ))
        self.assertEqual(data_samples[0].pred_label.score.shape, (2, ))
        torch.testing.assert_allclose(data_samples[0].pred_label.score,
                                      predictions[0].pred_label.score)

    def test_forward(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]
        model = MODELS.build(self.DEFAULT_ARGS)

        # test pure forward
        outs = model(inputs)
        # assert False, type(outs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        self.assertIsInstance(outs[0], torch.Tensor)

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
        model = MODELS.build(cfg)

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
        model = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 224, 224)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        predictions = model.test_step(data)
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))

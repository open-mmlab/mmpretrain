# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from typing import Callable
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch
from mmengine import ConfigDict
from mmengine.dataset.utils import default_collate
from torch.utils.data import DataLoader, Dataset

from mmpretrain.datasets.transforms import PackClsInputs
from mmpretrain.registry import MODELS
from mmpretrain.structures import ClsDataSample


class ExampleDataset(Dataset):

    def __init__(self):
        self.metainfo = None
        self.pipe = PackClsInputs()

    def __getitem__(self, idx):
        results = dict(
            img=np.random.random((64, 64, 3)), meta=dict(sampleidx=idx))

        return self.pipe(results)

    def __len__(self):
        return 10


class TestImageToImageRetriever(TestCase):
    DEFAULT_ARGS = dict(
        type='ImageToImageRetriever',
        image_encoder=[
            dict(type='ResNet', depth=18, out_indices=(3, )),
            dict(type='GlobalAveragePooling'),
        ],
        head=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss')),
        prototype=torch.rand((10, 512)),
    )

    def test_initialize(self):
        # test error prototype type
        cfg = {**self.DEFAULT_ARGS, 'prototype': 5}
        with self.assertRaises(AssertionError):
            model = MODELS.build(cfg)

        # test prototype is tensor
        model = MODELS.build(self.DEFAULT_ARGS)
        self.assertEqual(type(model.prototype), torch.Tensor)
        self.assertFalse(model.prototype_inited)
        self.assertIsInstance(model.similarity_fn, Callable)
        self.assertEqual(model.topk, -1)

        # test prototype is str
        cfg = {**self.DEFAULT_ARGS, 'prototype': './proto.pth'}
        model = MODELS.build(cfg)
        self.assertEqual(type(model.prototype), str)

        # test prototype is dict
        lodaer = DataLoader(ExampleDataset())
        cfg = {**self.DEFAULT_ARGS, 'prototype': lodaer}
        model = MODELS.build(cfg)
        self.assertEqual(type(model.prototype), DataLoader)

        # test prototype is dataloader
        loader_cfg = dict(
            batch_size=16,
            num_workers=2,
            dataset=dict(
                type='CIFAR100',
                data_prefix='data/cifar100',
                test_mode=False,
                pipeline=[]),
            sampler=dict(type='DefaultSampler', shuffle=True),
            persistent_workers=True)
        cfg = {**self.DEFAULT_ARGS, 'prototype': loader_cfg}
        model = MODELS.build(cfg)
        self.assertEqual(type(model.prototype), dict)

        # test similarity function
        self.assertEqual(model.similarity, 'cosine_similarity')

        def fn(a, b):
            return a * b

        cfg = {**self.DEFAULT_ARGS, 'similarity_fn': fn}
        model = MODELS.build(cfg)
        self.assertEqual(model.similarity, fn)
        self.assertIsInstance(model.similarity_fn, Callable)

        # test set batch augmentation from train_cfg
        cfg = {
            **self.DEFAULT_ARGS, 'train_cfg':
            dict(augments=dict(
                type='Mixup',
                alpha=1.,
            ))
        }
        model = MODELS.build(cfg)

        self.assertIsNotNone(model.data_preprocessor.batch_augments)

        cfg = {**self.DEFAULT_ARGS, 'train_cfg': dict()}
        model = MODELS.build(cfg)
        self.assertIsNone(model.data_preprocessor.batch_augments)

    def test_extract_feat(self):
        inputs = torch.rand(1, 3, 64, 64)
        cfg = ConfigDict(self.DEFAULT_ARGS)
        model = MODELS.build(cfg)

        # test extract_feat
        feats = model.extract_feat(inputs)
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].shape, (1, 512))

    def test_loss(self):
        inputs = torch.rand(1, 3, 64, 64)
        data_samples = [ClsDataSample().set_gt_label(1)]

        model = MODELS.build(self.DEFAULT_ARGS)
        losses = model.loss(inputs, data_samples)
        self.assertGreater(losses['loss'].item(), 0)

    def test_prepare_prototype(self):
        tmpdir = tempfile.TemporaryDirectory()
        # tensor
        cfg = {**self.DEFAULT_ARGS}
        model = MODELS.build(cfg)
        model.prepare_prototype()
        self.assertEqual(type(model.prototype_vecs), torch.Tensor)
        self.assertEqual(model.prototype_vecs.shape, (10, 512))
        self.assertTrue(model.prototype_inited)

        # test dump prototype
        ori_proto_vecs = model.prototype_vecs
        save_path = os.path.join(tmpdir.name, 'proto.pth')
        model.dump_prototype(save_path)

        # Check whether the saved feature exists
        feat = torch.load(save_path)
        self.assertEqual(feat.shape, (10, 512))

        # str
        cfg = {**self.DEFAULT_ARGS, 'prototype': save_path}
        model = MODELS.build(cfg)
        model.prepare_prototype()
        self.assertEqual(type(model.prototype_vecs), torch.Tensor)
        self.assertEqual(model.prototype_vecs.shape, (10, 512))
        self.assertTrue(model.prototype_inited)
        torch.allclose(ori_proto_vecs, model.prototype_vecs)

        # dict
        lodaer = DataLoader(ExampleDataset(), collate_fn=default_collate)
        cfg = {**self.DEFAULT_ARGS, 'prototype': lodaer}
        model = MODELS.build(cfg)
        model.prepare_prototype()
        self.assertEqual(type(model.prototype_vecs), torch.Tensor)
        self.assertEqual(model.prototype_vecs.shape, (10, 512))
        self.assertTrue(model.prototype_inited)

        tmpdir.cleanup()

    def test_predict(self):
        inputs = torch.rand(1, 3, 64, 64)
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
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))

        predictions = model.predict(inputs, data_samples)
        assert predictions is data_samples
        self.assertEqual(data_samples[0].pred_label.score.shape, (10, ))

    def test_forward(self):
        inputs = torch.rand(1, 3, 64, 64)
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
        model = MODELS.build(cfg)

        data = {
            'inputs': torch.randint(0, 256, (1, 3, 64, 64)),
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
            'inputs': torch.randint(0, 256, (1, 3, 64, 64)),
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
            'inputs': torch.randint(0, 256, (1, 3, 64, 64)),
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }

        predictions = model.test_step(data)
        self.assertEqual(predictions[0].pred_label.score.shape, (10, ))

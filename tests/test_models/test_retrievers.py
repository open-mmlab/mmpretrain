# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from typing import Callable
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine import ConfigDict
from mmengine.runner import Runner
from PIL import Image
from torch.utils.data import DataLoader

from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from mmcls.utils import register_all_modules

register_all_modules()


class TestImageToImageRetriever(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Simulate when the prototype is a dataset,
        # take the CUB dataset as an example
        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.image_folder = 'images'
        cls.ann_file = 'images.txt'
        cls.image_class_labels_file = 'classes.txt'
        cls.train_test_split_file = 'split.txt'

        with open(osp.join(cls.root, cls.image_class_labels_file), 'w') as f:
            f.write('\n'.join([
                '1 1', '2 1', '3 1', '4 1', '5 1', '6 1', '7 1', '8 1', '9 1',
                '10 1'
            ]))

        with open(osp.join(cls.root, cls.train_test_split_file), 'w') as f:
            f.write('\n'.join([
                '1 0', '2 0', '3 0', '4 0', '5 0', '6 1', '7 1', '8 1', '9 1',
                '10 1'
            ]))

        with open(osp.join(cls.root, cls.ann_file), 'w') as f:
            f.write('\n'.join([
                '1 1.jpg', '2 2.jpg', '3 3.jpg', '4 4.jpg', '5 5.jpg',
                '6 6.jpg', '7 7.jpg', '8 8.jpg', '9 9.jpg', '10 10.jpg'
            ]))

        if not osp.exists(f'{cls.root}/{cls.image_folder}'):
            os.makedirs(f'{cls.root}/{cls.image_folder}')
        for i in range(10):
            image = Image.new('RGB', (250, 250), (0, 0, 0))
            image.save(f'{cls.root}/{cls.image_folder}/{i}.jpg')

        cls.fake_feat = torch.randn((10, 512))
        cls.feat_path = f'{cls.root}/feat.pth'
        torch.save(cls.fake_feat, cls.feat_path)
        cls.cub_dataloader = dict(
            batch_size=8,
            dataset=dict(
                type='CUB',
                data_root=cls.root,
                ann_file=cls.ann_file,
                test_mode=True,
                data_prefix=cls.image_folder,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='Resize', scale=(224, 224)),
                    dict(type='PackClsInputs')
                ],
                image_class_labels_file=cls.image_class_labels_file,
                train_test_split_file=cls.train_test_split_file),
            sampler=dict(type='DefaultSampler', shuffle=False))

        cls.DEFAULT_ARGS = dict(
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
            prototype=torch.randn((10, 512)))

    def test_initialize(self):
        model = MODELS.build(self.DEFAULT_ARGS)
        # test prototype is tensor
        self.assertEqual(type(model.prototype), torch.Tensor)
        self.assertFalse(model.prototype_inited)
        self.assertIsNone(model.prototype_vecs)
        self.assertIsInstance(model.similarity_fn, Callable)
        self.assertEqual(model.topk, -1)

        # test prototype is str
        cfg = {**self.DEFAULT_ARGS, 'prototype': self.feat_path}
        model = MODELS.build(cfg)
        self.assertEqual(type(model.prototype), str)

        # test prototype is dict
        cfg = {**self.DEFAULT_ARGS, 'prototype': self.cub_dataloader}
        model = MODELS.build(cfg)
        self.assertEqual(type(model.prototype), dict)

        # test prototype is dataloader
        data_loader = Runner.build_dataloader(self.cub_dataloader)
        cfg = {**self.DEFAULT_ARGS, 'prototype': data_loader}
        model = MODELS.build(cfg)
        self.assertEqual(type(model.prototype), DataLoader)

        # test similarity function
        self.assertEqual(model.similarity, 'cosine_similarity')

        def fn(a, b):
            return a * b

        cfg = {**self.DEFAULT_ARGS, 'similarity_fn': fn}
        model = MODELS.build(cfg)
        self.assertEqual(model.similarity, fn)
        self.assertIsInstance(model.similarity_fn, Callable)

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

        # test extract_feat
        feats = model.extract_feat(inputs)
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].shape, (1, 512))

    def test_loss(self):
        inputs = torch.rand(1, 3, 224, 224)
        data_samples = [ClsDataSample().set_gt_label(1)]

        model = MODELS.build(self.DEFAULT_ARGS)
        losses = model.loss(inputs, data_samples)
        self.assertGreater(losses['loss'].item(), 0)

    def test_prepare_prototype(self):
        # tensor
        cfg = {**self.DEFAULT_ARGS}
        model = MODELS.build(cfg)
        model.prepare_prototype()
        self.assertEqual(type(model.prototype_vecs), torch.Tensor)
        self.assertEqual(model.prototype_vecs.shape, (10, 512))
        self.assertTrue(model.prototype_inited)

        # str
        cfg = {**self.DEFAULT_ARGS, 'prototype': self.feat_path}
        model = MODELS.build(cfg)
        model.prepare_prototype()
        self.assertEqual(type(model.prototype_vecs), torch.Tensor)
        self.assertEqual(model.prototype_vecs.shape, (10, 512))
        self.assertTrue(model.prototype_inited)

        # dict
        cfg = {**self.DEFAULT_ARGS, 'prototype': self.cub_dataloader}
        model = MODELS.build(cfg)
        model.prepare_prototype()
        self.assertEqual(type(model.prototype_vecs), torch.Tensor)
        self.assertEqual(model.prototype_vecs.shape, (5, 512))
        self.assertTrue(model.prototype_inited)

        # dataloader
        data_loader = Runner.build_dataloader(self.cub_dataloader)
        cfg = {**self.DEFAULT_ARGS, 'prototype': data_loader}
        model = MODELS.build(cfg)
        model.prepare_prototype()
        self.assertEqual(type(model.prototype_vecs), torch.Tensor)
        self.assertEqual(model.prototype_vecs.shape, (5, 512))
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
        model = MODELS.build(cfg)

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

    def test_dump_prototype(self):
        cfg = ConfigDict(self.DEFAULT_ARGS)
        model = MODELS.build(cfg)
        self.assertIsNone(model.dump_prototype(self.feat_path))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

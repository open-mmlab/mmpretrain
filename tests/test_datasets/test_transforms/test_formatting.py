# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import mmcv
import numpy as np
import torch
from PIL import Image

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample, MultiTaskDataSample


class TestPackInputs(unittest.TestCase):

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/color.jpg')
        data = {
            'sample_idx': 1,
            'img_path': img_path,
            'ori_shape': (300, 400),
            'img_shape': (300, 400),
            'scale_factor': 1.0,
            'flip': False,
            'img': mmcv.imread(img_path),
            'gt_label': 2,
            'custom_key': torch.tensor([1, 2, 3])
        }

        cfg = dict(type='PackInputs', algorithm_keys=['custom_key'])
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], DataSample)
        self.assertIn('flip', results['data_samples'].metainfo_keys())
        self.assertIsInstance(results['data_samples'].gt_label, torch.Tensor)
        self.assertIsInstance(results['data_samples'].custom_key, torch.Tensor)

        # Test grayscale image
        data['img'] = data['img'].mean(-1)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertEqual(results['inputs'].shape, (1, 300, 400))

        # Test video input
        data['img'] = np.random.randint(
            0, 256, (10, 3, 1, 224, 224), dtype=np.uint8)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertEqual(results['inputs'].shape, (10, 3, 1, 224, 224))

        # Test Pillow input
        data['img'] = Image.open(img_path)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertEqual(results['inputs'].shape, (3, 300, 400))

        # Test without `img` and `gt_label`
        del data['img']
        del data['gt_label']
        results = transform(copy.deepcopy(data))
        self.assertNotIn('gt_label', results['data_samples'])

    def test_repr(self):
        cfg = dict(type='PackInputs', meta_keys=['flip', 'img_shape'])
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), "PackInputs(input_key='img', algorithm_keys=(), "
            "meta_keys=['flip', 'img_shape'])")


class TestTranspose(unittest.TestCase):

    def test_transform(self):
        cfg = dict(type='Transpose', keys=['img'], order=[2, 0, 1])
        transform = TRANSFORMS.build(cfg)

        data = {'img': np.random.randint(0, 256, (224, 224, 3), dtype='uint8')}

        results = transform(copy.deepcopy(data))
        self.assertEqual(results['img'].shape, (3, 224, 224))

    def test_repr(self):
        cfg = dict(type='Transpose', keys=['img'], order=(2, 0, 1))
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), "Transpose(keys=['img'], order=(2, 0, 1))")


class TestToPIL(unittest.TestCase):

    def test_transform(self):
        cfg = dict(type='ToPIL')
        transform = TRANSFORMS.build(cfg)

        data = {'img': np.random.randint(0, 256, (224, 224, 3), dtype='uint8')}

        results = transform(copy.deepcopy(data))
        self.assertIsInstance(results['img'], Image.Image)

        cfg = dict(type='ToPIL', to_rgb=True)
        transform = TRANSFORMS.build(cfg)

        data = {'img': np.random.randint(0, 256, (224, 224, 3), dtype='uint8')}

        results = transform(copy.deepcopy(data))
        self.assertIsInstance(results['img'], Image.Image)
        np.equal(np.array(results['img']), data['img'][:, :, ::-1])

    def test_repr(self):
        cfg = dict(type='ToPIL', to_rgb=True)
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(repr(transform), 'NumpyToPIL(to_rgb=True)')


class TestToNumpy(unittest.TestCase):

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/color.jpg')
        data = {
            'img': Image.open(img_path),
        }

        cfg = dict(type='ToNumpy')
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIsInstance(results['img'], np.ndarray)
        self.assertEqual(results['img'].dtype, 'uint8')

        cfg = dict(type='ToNumpy', to_bgr=True)
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIsInstance(results['img'], np.ndarray)
        self.assertEqual(results['img'].dtype, 'uint8')
        np.equal(results['img'], np.array(data['img'])[:, :, ::-1])

    def test_repr(self):
        cfg = dict(type='ToNumpy', to_bgr=True)
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), 'PILToNumpy(to_bgr=True, dtype=None)')


class TestCollect(unittest.TestCase):

    def test_transform(self):
        data = {'img': [1, 2, 3], 'gt_label': 1}

        cfg = dict(type='Collect', keys=['img'])
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIn('img', results)
        self.assertNotIn('gt_label', results)

    def test_repr(self):
        cfg = dict(type='Collect', keys=['img'])
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(repr(transform), "Collect(keys=['img'])")


class TestPackMultiTaskInputs(unittest.TestCase):

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/color.jpg')
        data = {
            'sample_idx': 1,
            'img_path': img_path,
            'ori_shape': (300, 400),
            'img_shape': (300, 400),
            'scale_factor': 1.0,
            'flip': False,
            'img': mmcv.imread(img_path),
            'gt_label': {
                'task1': 1,
                'task3': 3
            },
        }

        cfg = dict(type='PackMultiTaskInputs', multi_task_fields=['gt_label'])
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], MultiTaskDataSample)
        self.assertIn('flip', results['data_samples'].task1.metainfo_keys())
        self.assertIsInstance(results['data_samples'].task1.gt_label,
                              torch.Tensor)

        # Test grayscale image
        data['img'] = data['img'].mean(-1)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertEqual(results['inputs'].shape, (1, 300, 400))

        # Test without `img` and `gt_label`
        del data['img']
        del data['gt_label']
        results = transform(copy.deepcopy(data))
        self.assertNotIn('gt_label', results['data_samples'])

    def test_repr(self):
        cfg = dict(
            type='PackMultiTaskInputs',
            multi_task_fields=['gt_label'],
            task_handlers=dict(task1=dict(type='PackInputs')),
        )
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform),
            "PackMultiTaskInputs(multi_task_fields=['gt_label'], "
            "input_key='img', task_handlers={'task1': PackInputs})")

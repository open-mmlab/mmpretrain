# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import mmcv
import numpy as np
import torch
from mmengine.structures import LabelData
from PIL import Image

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import ClsDataSample, MultiTaskDataSample


class TestPackClsInputs(unittest.TestCase):

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
        }

        cfg = dict(type='PackClsInputs')
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], ClsDataSample)
        self.assertIn('flip', results['data_samples'].metainfo_keys())
        self.assertIsInstance(results['data_samples'].gt_label, LabelData)

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
        cfg = dict(type='PackClsInputs', meta_keys=['flip', 'img_shape'])
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), "PackClsInputs(meta_keys=['flip', 'img_shape'])")


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


class TestToNumpy(unittest.TestCase):

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), '../../data/color.jpg')
        data = {
            'tensor': torch.tensor([1, 2, 3]),
            'Image': Image.open(img_path),
        }

        cfg = dict(type='ToNumpy', keys=['tensor', 'Image'], dtype='uint8')
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIsInstance(results['tensor'], np.ndarray)
        self.assertEqual(results['tensor'].dtype, 'uint8')
        self.assertIsInstance(results['Image'], np.ndarray)
        self.assertEqual(results['Image'].dtype, 'uint8')

    def test_repr(self):
        cfg = dict(type='ToNumpy', keys=['img'], dtype='uint8')
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(repr(transform), "ToNumpy(keys=['img'], dtype=uint8)")


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

        cfg = dict(type='PackMultiTaskInputs', )
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], MultiTaskDataSample)
        self.assertIn('flip', results['data_samples'].task1.metainfo_keys())
        self.assertIsInstance(results['data_samples'].task1.gt_label,
                              LabelData)

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
        cfg = dict(type='PackMultiTaskInputs', meta_keys=['img_shape'])
        transform = TRANSFORMS.build(cfg)
        rep = 'PackMultiTaskInputs(task_handlers={},'
        rep += ' multi_task_fields=(\'gt_label\',),'
        rep += ' meta_keys=[\'img_shape\'])'
        self.assertEqual(repr(transform), rep)

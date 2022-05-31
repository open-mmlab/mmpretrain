# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import mmcls.datasets  # noqa: F401,F403
from mmcls.registry import TRANSFORMS


def construct_toy_data():
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    # image
    results['ori_img'] = img
    results['img'] = copy.deepcopy(img)
    results['ori_shape'] = img.shape
    results['img_shape'] = img.shape
    return results


class TestResizeEdge(TestCase):

    def test_transform(self):
        results = dict(img=np.random.randint(0, 256, (128, 256, 3), np.uint8))

        # test resize short edge by default.
        cfg = dict(type='ResizeEdge', scale=224)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 448, 3))

        # test resize long edge.
        cfg = dict(type='ResizeEdge', scale=224, edge='long')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (112, 224, 3))

        # test resize width.
        cfg = dict(type='ResizeEdge', scale=224, edge='width')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (112, 224, 3))

        # test resize height.
        cfg = dict(type='ResizeEdge', scale=224, edge='height')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 448, 3))

        # test invalid edge
        with self.assertRaisesRegex(AssertionError, 'Invalid edge "hi"'):
            cfg = dict(type='ResizeEdge', scale=224, edge='hi')
            TRANSFORMS.build(cfg)

    def test_repr(self):
        cfg = dict(type='ResizeEdge', scale=224, edge='height')
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), 'ResizeEdge(scale=224, edge=height, backend=cv2, '
            'interpolation=bilinear)')


class TestRandomErasing(TestCase):

    def test_initialize(self):
        # test erase_prob assertion
        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomErasing', erase_prob=-1.)
            TRANSFORMS.build(cfg)
        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomErasing', erase_prob=1)
            TRANSFORMS.build(cfg)

        # test area_ratio assertion
        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomErasing', min_area_ratio=-1.)
            TRANSFORMS.build(cfg)
        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomErasing', max_area_ratio=1)
            TRANSFORMS.build(cfg)
        with self.assertRaises(AssertionError):
            # min_area_ratio should be smaller than max_area_ratio
            cfg = dict(
                type='RandomErasing', min_area_ratio=0.6, max_area_ratio=0.4)
            TRANSFORMS.build(cfg)

        # test aspect_range assertion
        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomErasing', aspect_range='str')
            TRANSFORMS.build(cfg)
        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomErasing', aspect_range=-1)
            TRANSFORMS.build(cfg)
        with self.assertRaises(AssertionError):
            # In aspect_range (min, max), min should be smaller than max.
            cfg = dict(type='RandomErasing', aspect_range=[1.6, 0.6])
            TRANSFORMS.build(cfg)

        # test mode assertion
        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomErasing', mode='unknown')
            TRANSFORMS.build(cfg)

        # test fill_std assertion
        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomErasing', fill_std='unknown')
            TRANSFORMS.build(cfg)

        # test implicit conversion of aspect_range
        cfg = dict(type='RandomErasing', aspect_range=0.5)
        random_erasing = TRANSFORMS.build(cfg)
        assert random_erasing.aspect_range == (0.5, 2.)

        cfg = dict(type='RandomErasing', aspect_range=2.)
        random_erasing = TRANSFORMS.build(cfg)
        assert random_erasing.aspect_range == (0.5, 2.)

        # test implicit conversion of fill_color
        cfg = dict(type='RandomErasing', fill_color=15)
        random_erasing = TRANSFORMS.build(cfg)
        assert random_erasing.fill_color == [15, 15, 15]

        # test implicit conversion of fill_std
        cfg = dict(type='RandomErasing', fill_std=0.5)
        random_erasing = TRANSFORMS.build(cfg)
        assert random_erasing.fill_std == [0.5, 0.5, 0.5]

    def test_transform(self):
        # test when erase_prob=0.
        results = construct_toy_data()
        cfg = dict(
            type='RandomErasing',
            erase_prob=0.,
            mode='const',
            fill_color=(255, 255, 255))
        random_erasing = TRANSFORMS.build(cfg)
        results = random_erasing(results)
        np.testing.assert_array_equal(results['img'], results['ori_img'])

        # test mode 'const'
        results = construct_toy_data()
        cfg = dict(
            type='RandomErasing',
            erase_prob=1.,
            mode='const',
            fill_color=(255, 255, 255))
        with patch('numpy.random', np.random.RandomState(0)):
            random_erasing = TRANSFORMS.build(cfg)
            results = random_erasing(results)
            expect_out = np.array(
                [[1, 255, 3, 4], [5, 255, 7, 8], [9, 10, 11, 12]],
                dtype=np.uint8)
            expect_out = np.stack([expect_out] * 3, axis=-1)
            np.testing.assert_array_equal(results['img'], expect_out)

        # test mode 'rand' with normal distribution
        results = construct_toy_data()
        cfg = dict(type='RandomErasing', erase_prob=1., mode='rand')
        with patch('numpy.random', np.random.RandomState(0)):
            random_erasing = TRANSFORMS.build(cfg)
            results = random_erasing(results)
            expect_out = results['ori_img']
            expect_out[:2, 1] = [[159, 98, 76], [14, 69, 122]]
            np.testing.assert_array_equal(results['img'], expect_out)

        # test mode 'rand' with uniform distribution
        results = construct_toy_data()
        cfg = dict(
            type='RandomErasing',
            erase_prob=1.,
            mode='rand',
            fill_std=(10, 255, 0))
        with patch('numpy.random', np.random.RandomState(0)):
            random_erasing = TRANSFORMS.build(cfg)
            results = random_erasing(results)

            expect_out = results['ori_img']
            expect_out[:2, 1] = [[113, 255, 128], [126, 83, 128]]
            np.testing.assert_array_equal(results['img'], expect_out)

    def test_repr(self):
        cfg = dict(
            type='RandomErasing',
            erase_prob=0.5,
            mode='const',
            aspect_range=(0.3, 1.3),
            fill_color=(255, 255, 255))
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform),
            'RandomErasing(erase_prob=0.5, min_area_ratio=0.02, '
            'max_area_ratio=0.4, aspect_range=(0.3, 1.3), mode=const, '
            'fill_color=(255, 255, 255), fill_std=None)')

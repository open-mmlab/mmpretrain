# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

import mmcls.datasets  # noqa: F401,F403
from mmcls.registry import TRANSFORMS


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

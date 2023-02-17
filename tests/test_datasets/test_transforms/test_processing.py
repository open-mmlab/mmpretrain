# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from unittest import TestCase
from unittest.mock import ANY, call, patch

import albumentations
import mmengine
import numpy as np

from mmpretrain.registry import TRANSFORMS


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


class TestRandomCrop(TestCase):

    def test_assertion(self):
        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomCrop', crop_size=-1)
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomCrop', crop_size=(1, 2, 3))
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomCrop', crop_size=(1, -2))
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomCrop', crop_size=224, padding_mode='co')
            TRANSFORMS.build(cfg)

    def test_transform(self):
        results = dict(img=np.random.randint(0, 256, (256, 256, 3), np.uint8))

        # test random crop by default.
        cfg = dict(type='RandomCrop', crop_size=224)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test int padding and int pad_val.
        cfg = dict(
            type='RandomCrop', crop_size=(224, 224), padding=2, pad_val=1)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test int padding and sequence pad_val.
        cfg = dict(
            type='RandomCrop', crop_size=224, padding=2, pad_val=(0, 50, 0))
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test sequence padding.
        cfg = dict(type='RandomCrop', crop_size=224, padding=(2, 3, 4, 5))
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test pad_if_needed.
        cfg = dict(
            type='RandomCrop',
            crop_size=300,
            pad_if_needed=True,
            padding_mode='edge')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (300, 300, 3))

        # test large crop size.
        results = dict(img=np.random.randint(0, 256, (256, 256, 3), np.uint8))
        cfg = dict(type='RandomCrop', crop_size=300)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (256, 256, 3))

        # test equal size.
        results = dict(img=np.random.randint(0, 256, (256, 256, 3), np.uint8))
        cfg = dict(type='RandomCrop', crop_size=256)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (256, 256, 3))

    def test_repr(self):
        cfg = dict(type='RandomCrop', crop_size=224)
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), 'RandomCrop(crop_size=(224, 224), padding=None, '
            'pad_if_needed=False, pad_val=0, padding_mode=constant)')


class TestRandomResizedCrop(TestCase):

    def test_assertion(self):
        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomResizedCrop', scale=-1)
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomResizedCrop', scale=(1, 2, 3))
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomResizedCrop', scale=(1, -2))
            TRANSFORMS.build(cfg)

        with self.assertRaises(ValueError):
            cfg = dict(
                type='RandomResizedCrop', scale=224, crop_ratio_range=(1, 0.1))
            TRANSFORMS.build(cfg)

        with self.assertRaises(ValueError):
            cfg = dict(
                type='RandomResizedCrop',
                scale=224,
                aspect_ratio_range=(1, 0.1))
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomResizedCrop', scale=224, max_attempts=-1)
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(type='RandomResizedCrop', scale=224, interpolation='ne')
            TRANSFORMS.build(cfg)

    def test_transform(self):
        results = dict(img=np.random.randint(0, 256, (256, 256, 3), np.uint8))

        # test random crop by default.
        cfg = dict(type='RandomResizedCrop', scale=224)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test crop_ratio_range.
        cfg = dict(
            type='RandomResizedCrop',
            scale=(224, 224),
            crop_ratio_range=(0.5, 0.8))
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test aspect_ratio_range.
        cfg = dict(
            type='RandomResizedCrop', scale=224, aspect_ratio_range=(0.5, 0.8))
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test max_attempts.
        cfg = dict(type='RandomResizedCrop', scale=224, max_attempts=0)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))
        # test fall back with extreme low in_ratio
        results = dict(img=np.random.randint(0, 256, (10, 256, 3), np.uint8))
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))
        # test fall back with extreme low in_ratio
        results = dict(img=np.random.randint(0, 256, (256, 10, 3), np.uint8))
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test large crop size.
        results = dict(img=np.random.randint(0, 256, (256, 256, 3), np.uint8))
        cfg = dict(type='RandomResizedCrop', scale=300)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (300, 300, 3))

    def test_repr(self):
        cfg = dict(type='RandomResizedCrop', scale=224)
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), 'RandomResizedCrop(scale=(224, 224), '
            'crop_ratio_range=(0.08, 1.0), aspect_ratio_range=(0.75, 1.3333), '
            'max_attempts=10, interpolation=bilinear, backend=cv2)')


class TestEfficientNetRandomCrop(TestCase):

    def test_assertion(self):
        with self.assertRaises(AssertionError):
            cfg = dict(type='EfficientNetRandomCrop', scale=(1, 1))
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(
                type='EfficientNetRandomCrop', scale=224, min_covered=-1)
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(
                type='EfficientNetRandomCrop', scale=224, crop_padding=-1)
            TRANSFORMS.build(cfg)

    def test_transform(self):
        results = dict(img=np.random.randint(0, 256, (256, 256, 3), np.uint8))

        # test random crop by default.
        cfg = dict(type='EfficientNetRandomCrop', scale=224)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test crop_ratio_range.
        cfg = dict(
            type='EfficientNetRandomCrop',
            scale=224,
            crop_ratio_range=(0.5, 0.8))
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test aspect_ratio_range.
        cfg = dict(
            type='EfficientNetRandomCrop',
            scale=224,
            aspect_ratio_range=(0.5, 0.8))
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test max_attempts.
        cfg = dict(type='EfficientNetRandomCrop', scale=224, max_attempts=0)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test min_covered.
        cfg = dict(type='EfficientNetRandomCrop', scale=224, min_covered=.9)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test crop_padding.
        cfg = dict(
            type='EfficientNetRandomCrop',
            scale=224,
            min_covered=0.9,
            crop_padding=10)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test large crop size.
        results = dict(img=np.random.randint(0, 256, (256, 256, 3), np.uint8))
        cfg = dict(type='EfficientNetRandomCrop', scale=300)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (300, 300, 3))

    def test_repr(self):
        cfg = dict(type='EfficientNetRandomCrop', scale=224)
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), 'EfficientNetRandomCrop(scale=(224, 224), '
            'crop_ratio_range=(0.08, 1.0), aspect_ratio_range=(0.75, 1.3333), '
            'max_attempts=10, interpolation=bicubic, backend=cv2, '
            'min_covered=0.1, crop_padding=32)')


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


class TestEfficientNetCenterCrop(TestCase):

    def test_assertion(self):
        with self.assertRaises(AssertionError):
            cfg = dict(type='EfficientNetCenterCrop', crop_size=(1, 1))
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(type='EfficientNetCenterCrop', crop_size=-1)
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = dict(
                type='EfficientNetCenterCrop', crop_size=224, crop_padding=-1)
            TRANSFORMS.build(cfg)

    def test_transform(self):
        results = dict(img=np.random.randint(0, 256, (256, 256, 3), np.uint8))

        # test random crop by default.
        cfg = dict(type='EfficientNetCenterCrop', crop_size=224)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test crop_padding.
        cfg = dict(
            type='EfficientNetCenterCrop', crop_size=224, crop_padding=10)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (224, 224, 3))

        # test large crop size.
        results = dict(img=np.random.randint(0, 256, (256, 256, 3), np.uint8))
        cfg = dict(type='EfficientNetCenterCrop', crop_size=300)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertTupleEqual(results['img'].shape, (300, 300, 3))

    def test_repr(self):
        cfg = dict(type='EfficientNetCenterCrop', crop_size=224)
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), 'EfficientNetCenterCrop(crop_size=224, '
            'crop_padding=32, interpolation=bicubic, backend=cv2)')


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


class TestColorJitter(TestCase):

    DEFAULT_ARGS = dict(
        type='ColorJitter',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.2)

    def test_initialize(self):
        cfg = dict(
            type='ColorJitter',
            brightness=(0.8, 1.2),
            contrast=[0.5, 1.5],
            saturation=0.,
            hue=0.2)
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(transform.brightness, (0.8, 1.2))
        self.assertEqual(transform.contrast, (0.5, 1.5))
        self.assertIsNone(transform.saturation)
        self.assertEqual(transform.hue, (-0.2, 0.2))

        with self.assertRaisesRegex(ValueError, 'If hue is a single number'):
            cfg = {**self.DEFAULT_ARGS, 'hue': -0.2}
            TRANSFORMS.build(cfg)

        with self.assertRaisesRegex(TypeError, 'hue should be a single'):
            cfg = {**self.DEFAULT_ARGS, 'hue': [0.5, 0.4, 0.2]}
            TRANSFORMS.build(cfg)

        logger = mmengine.MMLogger.get_current_instance()
        with self.assertLogs(logger, 'WARN') as log:
            cfg = {**self.DEFAULT_ARGS, 'hue': [-1, 0.4]}
            transform = TRANSFORMS.build(cfg)
        self.assertIn('ColorJitter hue values', log.output[0])
        self.assertEqual(transform.hue, (-0.5, 0.4))

    def test_transform(self):
        ori_img = np.random.randint(0, 256, (256, 256, 3), np.uint8)
        results = dict(img=copy.deepcopy(ori_img))

        # test transform
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertEqual(results['img'].dtype, ori_img.dtype)
        assert not np.equal(results['img'], ori_img).all()

        # test call with brightness, contrast and saturation are all 0
        results = dict(img=copy.deepcopy(ori_img))
        cfg = dict(
            type='ColorJitter', brightness=0., contrast=0., saturation=0.)
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertEqual(results['img'].dtype, ori_img.dtype)
        assert np.equal(results['img'], ori_img).all()

        # test call index
        cfg = {**self.DEFAULT_ARGS, 'contrast': 0.}
        transform = TRANSFORMS.build(cfg)
        with patch('numpy.random', np.random.RandomState(0)):
            mmcv_module = 'mmpretrain.datasets.transforms.processing.mmcv'
            call_list = [
                call.adjust_color(ANY, alpha=ANY),
                call.adjust_hue(ANY, ANY),
                call.adjust_brightness(ANY, ANY)
            ]
            with patch(mmcv_module, autospec=True) as mock:
                transform(results)
                self.assertEqual(mock.mock_calls, call_list)

    def test_repr(self):
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), 'ColorJitter(brightness=(0.5, 1.5), '
            'contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.2, 0.2))')


class TestLighting(TestCase):

    def setUp(self):
        EIGVAL = [0.2175, 0.0188, 0.0045]
        EIGVEC = [
            [-0.5836, -0.6948, 0.4203],
            [-0.5808, -0.0045, -0.814],
            [-0.5675, 0.7192, 0.4009],
        ]
        self.DEFAULT_ARGS = dict(
            type='Lighting',
            eigval=EIGVAL,
            eigvec=EIGVEC,
            alphastd=25.5,
            to_rgb=False)

    def test_assertion(self):
        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['eigval'] = -1
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['eigvec'] = None
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['alphastd'] = 'Lighting'
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['eigvec'] = dict()
            TRANSFORMS.build(cfg)

        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['eigvec'] = [
                [-0.5836, -0.6948, 0.4203],
                [-0.5808, -0.0045, -0.814],
                [-0.5675, 0.7192, 0.4009, 0.10],
            ]
            TRANSFORMS.build(cfg)

    def test_transform(self):
        ori_img = np.ones((256, 256, 3), np.uint8) * 127
        results = dict(img=copy.deepcopy(ori_img))

        # Test transform with non-img-keyword result
        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            lightening_module = TRANSFORMS.build(cfg)
            empty_results = dict()
            lightening_module(empty_results)

        # test call
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        lightening_module = TRANSFORMS.build(cfg)
        with patch('numpy.random', np.random.RandomState(0)):
            results = lightening_module(results)
            self.assertEqual(results['img'].dtype, ori_img.dtype)
            assert not np.equal(results['img'], ori_img).all()

        # test call with alphastd == 0
        results = dict(img=copy.deepcopy(ori_img))
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['alphastd'] = 0.0
        lightening_module = TRANSFORMS.build(cfg)
        results = lightening_module(results)
        self.assertEqual(results['img'].dtype, ori_img.dtype)
        assert np.equal(results['img'], ori_img).all()

    def test_repr(self):
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), 'Lighting(eigval=[0.2175, 0.0188, 0.0045], eigvec'
            '=[[-0.5836, -0.6948, 0.4203], [-0.5808, -0.0045, -0.814], ['
            '-0.5675, 0.7192, 0.4009]], alphastd=25.5, to_rgb=False)')


class TestAlbumentations(TestCase):
    DEFAULT_ARGS = dict(
        type='Albumentations', transforms=[dict(type='ChannelShuffle', p=1)])

    def test_assertion(self):
        # Test with non-list transforms
        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['transforms'] = 1
            TRANSFORMS.build(cfg)

        # Test with non-dict transforms item.
        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['transforms'] = [dict(p=1)]
            TRANSFORMS.build(cfg)

        # Test with dict transforms item without keyword 'type'.
        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['transforms'] = [[]]
            TRANSFORMS.build(cfg)

        # Test with dict transforms item with wrong type.
        with self.assertRaises(TypeError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['transforms'] = [dict(type=[])]
            TRANSFORMS.build(cfg)

        # Test with dict transforms item with wrong type.
        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            cfg['keymap'] = []
            TRANSFORMS.build(cfg)

    def test_transform(self):
        ori_img = np.random.randint(0, 256, (256, 256, 3), np.uint8)
        results = dict(img=copy.deepcopy(ori_img))

        # Test transform with non-img-keyword result
        with self.assertRaises(AssertionError):
            cfg = copy.deepcopy(self.DEFAULT_ARGS)
            albu_module = TRANSFORMS.build(cfg)
            empty_results = dict()
            albu_module(empty_results)

        # Test normal case
        results = dict(img=copy.deepcopy(ori_img))
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        albu_module = TRANSFORMS.build(cfg)
        ablu_result = albu_module(results)

        # Test using 'Albu'
        results = dict(img=copy.deepcopy(ori_img))
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['type'] = 'Albu'
        albu_module = TRANSFORMS.build(cfg)
        ablu_result = albu_module(results)

        # Test with keymap
        results = dict(img=copy.deepcopy(ori_img))
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['keymap'] = dict(img='image')
        albu_module = TRANSFORMS.build(cfg)
        ablu_result = albu_module(results)

        # Test with nested transform
        results = dict(img=copy.deepcopy(ori_img))
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        nested_transform_cfg = [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
        cfg['transforms'] = nested_transform_cfg
        mmpretrain_module = TRANSFORMS.build(cfg)
        mmpretrain_module(results)

        # test to be same with albumentations 3rd package
        np.random.seed(0)
        random.seed(0)
        import albumentations as A
        ablu_transform_3rd = A.Compose([
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])
        transformed_image_3rd = ablu_transform_3rd(
            image=copy.deepcopy(ori_img))['image']

        np.random.seed(0)
        random.seed(0)
        results = dict(img=copy.deepcopy(ori_img))
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['transforms'] = [
            dict(type='RandomCrop', width=256, height=256),
            dict(type='HorizontalFlip', p=0.5),
            dict(type='RandomBrightnessContrast', p=0.2)
        ]
        mmpretrain_module = TRANSFORMS.build(cfg)
        transformed_image_mmpretrain = mmpretrain_module(results)['img']
        assert np.equal(transformed_image_3rd,
                        transformed_image_mmpretrain).all()

        # Test class obj case
        results = dict(img=np.random.randint(0, 256, (200, 300, 3), np.uint8))
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['transforms'] = [
            dict(type=albumentations.SmallestMaxSize, max_size=400, p=1)
        ]
        albu_module = TRANSFORMS.build(cfg)
        ablu_result = albu_module(results)
        assert 'img' in ablu_result
        assert min(ablu_result['img'].shape[:2]) == 400
        assert ablu_result['img_shape'] == (400, 600)

    def test_repr(self):
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(
            repr(transform), "Albumentations(transforms=[{'type': "
            "'ChannelShuffle', 'p': 1}])")

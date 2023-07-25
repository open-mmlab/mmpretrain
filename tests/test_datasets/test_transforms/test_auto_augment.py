# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from unittest import TestCase
from unittest.mock import ANY, patch

import numpy as np

from mmpretrain.registry import TRANSFORMS


def construct_toy_data():
    img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
    results = dict()
    # image
    results['ori_img'] = img
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']
    return results


def construct_toy_data_photometric():
    img = np.array([[0, 128, 255], [1, 127, 254], [2, 129, 253]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    # image
    results['ori_img'] = img
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']
    return results


class TestAutoAugment(TestCase):

    def test_construct(self):
        policies = [[
            dict(type='Posterize', bits=4, prob=0.4),
            dict(type='Rotate', angle=30., prob=0.6)
        ]]

        cfg = dict(type='AutoAugment', policies=policies)
        transform = TRANSFORMS.build(cfg)
        results = construct_toy_data()
        with patch.object(transform.transforms[0], 'transform') as mock:
            transform(results)
            mock.assert_called_once()

        cfg = dict(type='AutoAugment', policies='imagenet')
        transform = TRANSFORMS.build(cfg)
        with patch.object(transform.transforms[5], 'transform') as mock:
            with patch('numpy.random', np.random.RandomState(1)):
                transform(results)
                mock.assert_called()

        # test hparams
        cfg = dict(
            type='AutoAugment',
            policies=policies,
            hparams=dict(pad_val=[255, 255, 255]))
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(transform.policies[0][1]['pad_val'], [255, 255, 255])
        self.assertNotIn('pad_val', transform.policies[0][0])

        with self.assertRaisesRegex(AssertionError, 'choose from .*imagenet'):
            cfg = dict(type='AutoAugment', policies='unknown')
            transform = TRANSFORMS.build(cfg)

    def test_repr(self):
        policies = [[
            dict(type='Posterize', bits=4, prob=0.4),
            dict(type='Rotate', angle=30., prob=0.6)
        ]]

        cfg = dict(type='AutoAugment', policies=policies)
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Posterize, \tRotate', repr(transform))


class TestRandAugment(TestCase):
    DEFAULT_ARGS = dict(
        type='RandAugment',
        magnitude_level=7,
        num_policies=1,
        policies='timm_increasing')

    def test_construct(self):
        policies = [
            dict(type='Posterize', magnitude_range=(4, 0)),
            dict(type='Rotate', magnitude_range=(0, 30))
        ]

        cfg = {**self.DEFAULT_ARGS, 'policies': policies}
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(len(list(transform)), 2)
        results = construct_toy_data()
        with patch.object(transform.transforms[1], 'transform') as mock:
            with patch('numpy.random', np.random.RandomState(1)):
                transform(results)
                mock.assert_called_once()

        cfg = {**self.DEFAULT_ARGS, 'policies': 'timm_increasing'}
        transform = TRANSFORMS.build(cfg)
        with patch.object(transform.transforms[5], 'transform') as mock:
            with patch('numpy.random', np.random.RandomState(1)):
                transform(results)
                mock.assert_called()

        # test hparams
        cfg = {
            **self.DEFAULT_ARGS,
            'policies': policies,
            'hparams': dict(pad_val=[255, 255, 255]),
        }
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(transform.policies[1]['pad_val'], [255, 255, 255])
        self.assertNotIn('pad_val', transform.policies[0])

        # test magnitude related parameters
        cfg = {
            **self.DEFAULT_ARGS, 'policies': [
                dict(type='Equalize'),
                dict(type='Rotate', magnitude_range=(0, 30))
            ]
        }
        transform = TRANSFORMS.build(cfg)
        self.assertNotIn('magnitude_range', transform.policies[0])
        self.assertNotIn('magnitude_level', transform.policies[0])
        self.assertNotIn('magnitude_range', transform.policies[0])
        self.assertNotIn('total_level', transform.policies[0])
        self.assertEqual(transform.policies[1]['magnitude_range'], (0, 30))
        self.assertEqual(transform.policies[1]['magnitude_level'], 7)
        self.assertEqual(transform.policies[1]['magnitude_std'], 0.)
        self.assertEqual(transform.policies[1]['total_level'], 10)

        # test invalid policies
        with self.assertRaisesRegex(AssertionError,
                                    'choose from .*timm_increasing'):
            cfg = {**self.DEFAULT_ARGS, 'policies': 'unknown'}
            transform = TRANSFORMS.build(cfg)

        # test invalid magnitude_std
        with self.assertRaisesRegex(AssertionError, 'got "unknown" instead'):
            cfg = {**self.DEFAULT_ARGS, 'magnitude_std': 'unknown'}
            transform = TRANSFORMS.build(cfg)

    def test_repr(self):
        policies = [
            dict(type='Posterize', magnitude_range=(4, 0)),
            dict(type='Equalize')
        ]

        cfg = {**self.DEFAULT_ARGS, 'policies': policies}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('    Posterize (4, 0)\n    Equalize\n', repr(transform))


class TestShear(TestCase):
    DEFAULT_ARGS = dict(type='Shear')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {
                **self.DEFAULT_ARGS, 'magnitude': 1,
                'magnitude_range': (1, 2)
            }
            TRANSFORMS.build(cfg)

        with self.assertRaisesRegex(AssertionError, 'got "unknown" instead'):
            cfg = {**self.DEFAULT_ARGS, 'magnitude': 1, 'direction': 'unknown'}
            TRANSFORMS.build(cfg)

    def test_transform(self):
        # test params inputs
        with patch('mmcv.imshear') as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.2,
                'random_negative_prob': 0.,
                'prob': 1.,
                'direction': 'horizontal',
                'pad_val': 255,
                'interpolation': 'nearest',
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                0.2,
                direction='horizontal',
                border_value=255,
                interpolation='nearest')

        # test random_negative_prob
        with patch('mmcv.imshear') as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.2,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY, -0.2, direction=ANY, border_value=ANY, interpolation=ANY)

        # test prob
        with patch('mmcv.imshear') as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.2,
                'random_negative_prob': 0.,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test sequeue pad_val
        with patch('mmcv.imshear') as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.2,
                'random_negative_prob': 0.,
                'prob': 1.,
                'direction': 'horizontal',
                'pad_val': (255, 255, 255),
                'interpolation': 'nearest',
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                0.2,
                direction='horizontal',
                border_value=(255, 255, 255),
                interpolation='nearest')

        # test magnitude_range
        with patch('mmcv.imshear') as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 0.3),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY, 0.18, direction=ANY, border_value=ANY, interpolation=ANY)

        # test magnitude_std is positive
        with patch('mmcv.imshear') as mock:
            cfg = {
                **self.DEFAULT_ARGS, 'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 0.3),
                'magnitude_std': 1
            }
            with patch('numpy.random', np.random.RandomState(1)):
                TRANSFORMS.build(cfg)(construct_toy_data())
                self.assertAlmostEqual(mock.call_args[0][1], 0.1811, places=4)

        # test magnitude_std = 'inf'
        with patch('mmcv.imshear') as mock:
            cfg = {
                **self.DEFAULT_ARGS, 'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 0.3),
                'magnitude_std': 'inf'
            }
            with patch('numpy.random', np.random.RandomState(9)):
                TRANSFORMS.build(cfg)(construct_toy_data())
                self.assertAlmostEqual(mock.call_args[0][1], 0.0882, places=4)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'magnitude': 0.1}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Shear(magnitude=0.1', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 0.3)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Shear(magnitude=None', repr(transform))
        self.assertIn('magnitude_range=(0, 0.3)', repr(transform))


class TestTranslate(TestCase):
    DEFAULT_ARGS = dict(type='Translate')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {
                **self.DEFAULT_ARGS, 'magnitude': 1,
                'magnitude_range': (1, 2)
            }
            TRANSFORMS.build(cfg)

        with self.assertRaisesRegex(AssertionError, 'got "unknown" instead'):
            cfg = {**self.DEFAULT_ARGS, 'magnitude': 1, 'direction': 'unknown'}
            TRANSFORMS.build(cfg)

    def test_transform(self):
        transform_func = 'mmcv.imtranslate'

        # test params inputs
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.2,
                'random_negative_prob': 0.,
                'prob': 1.,
                'direction': 'horizontal',
                'pad_val': 255,
                'interpolation': 'nearest',
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                200 * 0.2,
                direction='horizontal',
                border_value=255,
                interpolation='nearest')

        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.2,
                'random_negative_prob': 0.,
                'prob': 1.,
                'direction': 'vertical',
                'pad_val': 255,
                'interpolation': 'nearest',
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                100 * 0.2,
                direction='vertical',
                border_value=255,
                interpolation='nearest')

        # test sequeue pad_val
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.2,
                'random_negative_prob': 0.,
                'prob': 1.,
                'direction': 'horizontal',
                'pad_val': [255, 255, 255],
                'interpolation': 'nearest',
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                200 * 0.2,
                direction='horizontal',
                border_value=(255, 255, 255),
                interpolation='nearest')

        # test prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.2,
                'random_negative_prob': 0.,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.2,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                -0.2 * 200,
                direction=ANY,
                border_value=ANY,
                interpolation=ANY)

        # test magnitude_range
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 0.3),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                0.18 * 200,
                direction=ANY,
                border_value=ANY,
                interpolation=ANY)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'magnitude': 0.1}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Translate(magnitude=0.1', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 0.3)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Translate(magnitude=None', repr(transform))
        self.assertIn('magnitude_range=(0, 0.3)', repr(transform))


class TestRotate(TestCase):
    DEFAULT_ARGS = dict(type='Rotate')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {**self.DEFAULT_ARGS, 'angle': 30, 'magnitude_range': (1, 2)}
            TRANSFORMS.build(cfg)

    def test_transform(self):
        transform_func = 'mmcv.imrotate'

        # test params inputs
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'angle': 30,
                'center': (10, 10),
                'random_negative_prob': 0.,
                'prob': 1.,
                'scale': 1.5,
                'pad_val': 255,
                'interpolation': 'bilinear',
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                30,
                center=(10, 10),
                scale=1.5,
                border_value=255,
                interpolation='bilinear')

        # test params inputs
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'angle': 30,
                'center': (10, 10),
                'random_negative_prob': 0.,
                'prob': 1.,
                'scale': 1.5,
                'pad_val': (255, 255, 255),
                'interpolation': 'bilinear',
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                30,
                center=(10, 10),
                scale=1.5,
                border_value=(255, 255, 255),
                interpolation='bilinear')

        # test prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'angle': 30,
                'random_negative_prob': 0.,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'angle': 30,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                -30,
                center=ANY,
                scale=ANY,
                border_value=ANY,
                interpolation=ANY)

        # test magnitude_range
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 30),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY,
                18,
                center=ANY,
                scale=ANY,
                border_value=ANY,
                interpolation=ANY)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'angle': 30}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Rotate(angle=30', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 30)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Rotate(angle=None', repr(transform))
        self.assertIn('magnitude_range=(0, 30)', repr(transform))


class TestAutoContrast(TestCase):
    DEFAULT_ARGS = dict(type='AutoContrast')

    def test_transform(self):
        transform_func = 'mmcv.auto_contrast'

        # test prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        # No effect
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY)

        # test magnitude_range
        # No effect
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 30),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'prob': 0.5}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('AutoContrast(prob=0.5)', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 30)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('AutoContrast(prob=', repr(transform))
        self.assertNotIn('magnitude_range=(0, 30)', repr(transform))


class TestInvert(TestCase):
    DEFAULT_ARGS = dict(type='Invert')

    def test_transform(self):
        transform_func = 'mmcv.iminvert'

        # test prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        # No effect
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY)

        # test magnitude_range
        # No effect
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 30),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'prob': 0.5}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Invert(prob=0.5)', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 30)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Invert(prob=', repr(transform))
        self.assertNotIn('magnitude_range=(0, 30)', repr(transform))


class TestEqualize(TestCase):
    DEFAULT_ARGS = dict(type='Equalize')

    def test_transform(self):
        transform_func = 'mmcv.imequalize'

        # test prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        # No effect
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY)

        # test magnitude_range
        # No effect
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 30),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'prob': 0.5}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Equalize(prob=0.5)', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 30)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Equalize(prob=', repr(transform))
        self.assertNotIn('magnitude_range=(0, 30)', repr(transform))


class TestSolarize(TestCase):
    DEFAULT_ARGS = dict(type='Solarize')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {**self.DEFAULT_ARGS, 'thr': 1, 'magnitude_range': (1, 2)}
            TRANSFORMS.build(cfg)

    def test_transform(self):
        transform_func = 'mmcv.solarize'

        # test params inputs
        with patch(transform_func) as mock:
            cfg = {**self.DEFAULT_ARGS, 'thr': 128, 'prob': 1.}
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, thr=128)

        # test prob
        with patch(transform_func) as mock:
            cfg = {**self.DEFAULT_ARGS, 'thr': 128, 'prob': 0.}
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        # cannot accept `random_negative_prob` argument
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'thr': 128,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            with self.assertRaisesRegex(TypeError, 'multiple values'):
                TRANSFORMS.build(cfg)(construct_toy_data())

        # test magnitude_range
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (256, 0),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, thr=256 * 0.4)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'thr': 128}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Solarize(thr=128', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (256, 0)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Solarize(thr=None', repr(transform))
        self.assertIn('magnitude_range=(256, 0)', repr(transform))


class TestSolarizeAdd(TestCase):
    DEFAULT_ARGS = dict(type='SolarizeAdd')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {
                **self.DEFAULT_ARGS, 'magnitude': 1,
                'magnitude_range': (1, 2)
            }
            TRANSFORMS.build(cfg)

        with self.assertRaisesRegex(AssertionError, 'str'):
            cfg = {**self.DEFAULT_ARGS, 'magnitude': 1, 'thr': 'hi'}
            TRANSFORMS.build(cfg)

    def test_transform(self):

        # test params inputs
        cfg = {**self.DEFAULT_ARGS, 'magnitude': 100, 'thr': 128, 'prob': 1.}
        results = construct_toy_data_photometric()
        expected = np.where(results['img'] < 128,
                            np.minimum(results['img'] + 100, 255),
                            results['img'])
        TRANSFORMS.build(cfg)(results)
        np.testing.assert_allclose(results['img'], expected)

        # test prob
        cfg = {**self.DEFAULT_ARGS, 'magnitude': 100, 'thr': 128, 'prob': 0.}
        results = construct_toy_data_photometric()
        expected = copy.deepcopy(results['img'])
        TRANSFORMS.build(cfg)(results)
        np.testing.assert_allclose(results['img'], expected)

        # test random_negative_prob
        # cannot accept `random_negative_prob` argument
        cfg = {
            **self.DEFAULT_ARGS,
            'magnitude': 100,
            'thr': 128,
            'random_negative_prob': 1.,
            'prob': 1.,
        }
        with self.assertRaisesRegex(TypeError, 'multiple values'):
            TRANSFORMS.build(cfg)(construct_toy_data())

        # test magnitude_range
        cfg = {
            **self.DEFAULT_ARGS,
            'prob': 1.,
            'magnitude_level': 6,
            'magnitude_range': (0, 110),
        }
        results = construct_toy_data_photometric()
        expected = np.where(results['img'] < 128,
                            np.minimum(results['img'] + 110 * 0.6, 255),
                            results['img'])
        TRANSFORMS.build(cfg)(results)
        np.testing.assert_allclose(results['img'], expected)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'magnitude': 100}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('SolarizeAdd(magnitude=100', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 110)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('SolarizeAdd(magnitude=None', repr(transform))
        self.assertIn('magnitude_range=(0, 110)', repr(transform))


class TestPosterize(TestCase):
    DEFAULT_ARGS = dict(type='Posterize')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {**self.DEFAULT_ARGS, 'bits': 1, 'magnitude_range': (1, 2)}
            TRANSFORMS.build(cfg)

        with self.assertRaisesRegex(AssertionError, 'got 100 instead'):
            cfg = {**self.DEFAULT_ARGS, 'bits': 100}
            TRANSFORMS.build(cfg)

    def test_transform(self):
        transform_func = 'mmcv.posterize'

        # test params inputs
        with patch(transform_func) as mock:
            cfg = {**self.DEFAULT_ARGS, 'bits': 4, 'prob': 1.}
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, bits=4)

        # test prob
        with patch(transform_func) as mock:
            cfg = {**self.DEFAULT_ARGS, 'bits': 4, 'prob': 0.}
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        # cannot accept `random_negative_prob` argument
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'bits': 4,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            with self.assertRaisesRegex(TypeError, 'multiple values'):
                TRANSFORMS.build(cfg)(construct_toy_data())

        # test magnitude_range
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (4, 0),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, bits=math.ceil(4 * 0.4))

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'bits': 4}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Posterize(bits=4', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (4, 0)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Posterize(bits=None', repr(transform))
        self.assertIn('magnitude_range=(4, 0)', repr(transform))


class TestContrast(TestCase):
    DEFAULT_ARGS = dict(type='Contrast')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {
                **self.DEFAULT_ARGS, 'magnitude': 1,
                'magnitude_range': (1, 2)
            }
            TRANSFORMS.build(cfg)

    def test_transform(self):
        transform_func = 'mmcv.adjust_contrast'

        # test params inputs
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 0.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, factor=1 + 0.5)

        # test prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 0.,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, factor=1 - 0.5)

        # test magnitude_range
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 0.5),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, factor=1 + 0.6 * 0.5)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'magnitude': 0.1}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Contrast(magnitude=0.1', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 0.3)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Contrast(magnitude=None', repr(transform))
        self.assertIn('magnitude_range=(0, 0.3)', repr(transform))


class TestColorTransform(TestCase):
    DEFAULT_ARGS = dict(type='ColorTransform')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {
                **self.DEFAULT_ARGS, 'magnitude': 1,
                'magnitude_range': (1, 2)
            }
            TRANSFORMS.build(cfg)

    def test_transform(self):
        transform_func = 'mmcv.adjust_color'

        # test params inputs
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 0.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, alpha=1 + 0.5)

        # test prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 0.,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, alpha=1 - 0.5)

        # test magnitude_range
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 0.5),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, alpha=1 + 0.6 * 0.5)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'magnitude': 0.1}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('ColorTransform(magnitude=0.1', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 0.3)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('ColorTransform(magnitude=None', repr(transform))
        self.assertIn('magnitude_range=(0, 0.3)', repr(transform))


class TestBrightness(TestCase):
    DEFAULT_ARGS = dict(type='Brightness')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {
                **self.DEFAULT_ARGS, 'magnitude': 1,
                'magnitude_range': (1, 2)
            }
            TRANSFORMS.build(cfg)

    def test_transform(self):
        transform_func = 'mmcv.adjust_brightness'

        # test params inputs
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 0.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, factor=1 + 0.5)

        # test prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 0.,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, factor=1 - 0.5)

        # test magnitude_range
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 0.5),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, factor=1 + 0.6 * 0.5)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'magnitude': 0.1}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Brightness(magnitude=0.1', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 0.3)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Brightness(magnitude=None', repr(transform))
        self.assertIn('magnitude_range=(0, 0.3)', repr(transform))


class TestSharpness(TestCase):
    DEFAULT_ARGS = dict(type='Sharpness')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {
                **self.DEFAULT_ARGS, 'magnitude': 1,
                'magnitude_range': (1, 2)
            }
            TRANSFORMS.build(cfg)

    def test_transform(self):
        transform_func = 'mmcv.adjust_sharpness'

        # test params inputs
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 0.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, factor=1 + 0.5)

        # test prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 0.,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test random_negative_prob
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude': 0.5,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, factor=1 - 0.5)

        # test magnitude_range
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'random_negative_prob': 0.,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (0, 0.5),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, factor=1 + 0.6 * 0.5)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'magnitude': 0.1}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Sharpness(magnitude=0.1', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 0.3)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Sharpness(magnitude=None', repr(transform))
        self.assertIn('magnitude_range=(0, 0.3)', repr(transform))


class TestCutout(TestCase):
    DEFAULT_ARGS = dict(type='Cutout')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {
                **self.DEFAULT_ARGS, 'shape': 10,
                'magnitude_range': (10, 20)
            }
            TRANSFORMS.build(cfg)

    def test_transform(self):
        transform_func = 'mmcv.cutout'

        # test params inputs
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'shape': (10, 15),
                'prob': 1.,
                'pad_val': 255,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, (10, 15), pad_val=255)

        # test prob
        with patch(transform_func) as mock:
            cfg = {**self.DEFAULT_ARGS, 'shape': 10, 'prob': 0.}
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test sequeue pad_val
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'shape': (10, 15),
                'prob': 1.,
                'pad_val': [255, 255, 255],
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(
                ANY, (10, 15), pad_val=(255, 255, 255))

        # test random_negative_prob
        # cannot accept `random_negative_prob` argument
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'shape': 10,
                'random_negative_prob': 1.,
                'prob': 1.,
            }
            with self.assertRaisesRegex(TypeError, 'multiple values'):
                TRANSFORMS.build(cfg)(construct_toy_data())

        # test magnitude_range
        with patch(transform_func) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'prob': 1.,
                'magnitude_level': 6,
                'magnitude_range': (1, 41),
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(ANY, 40 * 0.6 + 1, pad_val=ANY)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'shape': 15}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Cutout(shape=15', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0, 41)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('Cutout(shape=None', repr(transform))
        self.assertIn('magnitude_range=(0, 41)', repr(transform))


class TestGaussianBlur(TestCase):
    DEFAULT_ARGS = dict(type='GaussianBlur')

    def test_initialize(self):
        with self.assertRaisesRegex(AssertionError, 'only one of'):
            TRANSFORMS.build(self.DEFAULT_ARGS)

        with self.assertRaisesRegex(AssertionError, 'only one of'):
            cfg = {**self.DEFAULT_ARGS, 'radius': 1, 'magnitude_range': (1, 2)}
            TRANSFORMS.build(cfg)

    def test_transform(self):
        transform_func = 'PIL.ImageFilter.GaussianBlur'
        from PIL.ImageFilter import GaussianBlur

        # test params inputs
        with patch(transform_func, wraps=GaussianBlur) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'radius': 0.5,
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_called_once_with(radius=0.5)

        # test prob
        with patch(transform_func, wraps=GaussianBlur) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'radius': 0.5,
                'prob': 0.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            mock.assert_not_called()

        # test magnitude_range
        with patch(transform_func, wraps=GaussianBlur) as mock:
            cfg = {
                **self.DEFAULT_ARGS,
                'magnitude_range': (0.1, 2),
                'magnitude_std': 'inf',
                'prob': 1.,
            }
            TRANSFORMS.build(cfg)(construct_toy_data())
            self.assertTrue(0.1 < mock.call_args[1]['radius'] < 2)

    def test_repr(self):
        cfg = {**self.DEFAULT_ARGS, 'radius': 0.1}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('GaussianBlur(radius=0.1, prob=0.5', repr(transform))
        self.assertNotIn('magnitude_range', repr(transform))

        cfg = {**self.DEFAULT_ARGS, 'magnitude_range': (0.1, 2)}
        transform = TRANSFORMS.build(cfg)
        self.assertIn('GaussianBlur(radius=None, prob=0.5', repr(transform))
        self.assertIn('magnitude_range=(0.1, 2)', repr(transform))

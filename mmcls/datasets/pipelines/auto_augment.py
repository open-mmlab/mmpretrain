import copy

import mmcv
import numpy as np

from ..builder import PIPELINES
from .compose import Compose


def random_negative(value, random_negative_prob):
    """Randomly negate value based on random_negative_prob."""
    return -value if np.random.rand() < random_negative_prob else value


@PIPELINES.register_module()
class AutoAugment(object):

    def __init__(self, policies):
        if isinstance(policies, str):
            if policies == 'ImageNetPolicy':
                policies = [
                    [
                        dict(type='Posterize', bits=4, prob=0.4),
                        dict(type='Rotate', angle=30., prob=0.6)
                    ],
                    [
                        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
                        dict(type='AutoContrast', prob=0.5)
                    ],
                    [
                        dict(type='Equalize', prob=0.8),
                        dict(type='Equalize', prob=0.6)
                    ],
                    [
                        dict(type='Posterize', bits=5, prob=0.6),
                        dict(type='Posterize', bits=5, prob=0.6)
                    ],
                    [
                        dict(type='Equalize', prob=0.4),
                        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
                    ],
                    [
                        dict(type='Equalize', prob=0.4),
                        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8)
                    ],
                    [
                        dict(type='Solarize', thr=256 / 9 * 6, prob=0.6),
                        dict(type='Equalize', prob=0.6)
                    ],
                    [
                        dict(type='Posterize', bits=6, prob=0.8),
                        dict(type='Equalize', prob=1.)
                    ],
                    [
                        dict(type='Rotate', angle=10., prob=0.2),
                        dict(type='Solarize', thr=256 / 9, prob=0.6)
                    ],
                    [
                        dict(type='Equalize', prob=0.6),
                        dict(type='Posterize', bits=5, prob=0.6)
                    ],
                    [
                        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),
                        dict(type='ColorTransform', magnitude=0., prob=0.4)
                    ],
                    [
                        dict(type='Rotate', angle=30., prob=0.4),
                        dict(type='Equalize', prob=0.6)
                    ],
                    [
                        dict(type='Equalize', prob=0.0),
                        dict(type='Equalize', prob=0.8)
                    ],
                    [
                        dict(type='Invert', prob=0.6),
                        dict(type='Equalize', prob=1.)
                    ],
                    [
                        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
                        dict(type='Contrast', magnitude=0.8, prob=1.)
                    ],
                    [
                        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),
                        dict(type='ColorTransform', magnitude=0.2, prob=1.)
                    ],
                    [
                        dict(type='ColorTransform', magnitude=0.8, prob=0.8),
                        dict(type='Solarize', thr=256 / 9 * 2, prob=0.8)
                    ],
                    [
                        dict(type='Sharpness', magnitude=0.7, prob=0.4),
                        dict(type='Invert', prob=0.6)
                    ],
                    [
                        dict(
                            type='Shear',
                            magnitude=0.3 / 9 * 5,
                            prob=0.6,
                            direction='horizontal'),
                        dict(type='Equalize', prob=1.)
                    ],
                    [
                        dict(type='ColorTransform', magnitude=0., prob=0.4),
                        dict(type='Equalize', prob=0.6)
                    ],
                    [
                        dict(type='Equalize', prob=0.4),
                        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
                    ],
                    [
                        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
                        dict(type='AutoContrast', prob=0.6)
                    ],
                    [
                        dict(type='Invert', prob=0.6),
                        dict(type='Equalize', prob=1.)
                    ],
                    [
                        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
                        dict(type='Contrast', magnitude=0.8, prob=1.)
                    ],
                    [
                        dict(type='Equalize', prob=0.8),
                        dict(type='Equalize', prob=0.6)
                    ],
                ]
            elif policies == 'Cifar10Policy':
                # TODO fill this part
                pass
            else:
                raise ValueError(f'Unsupported policy name {policies}.')
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0, \
                'Each policy in policies must be a non-empty list.'
            for augment in policy:
                assert isinstance(augment, dict) and 'type' in augment, \
                    'Each specific augmentation must be a dict with key' \
                    ' "type".'

        self.policies = copy.deepcopy(policies)
        self.sub_policy = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        sub_policy = np.random.choice(self.sub_policy)
        return sub_policy(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(policies={self.policies})'
        return repr_str


@PIPELINES.register_module()
class Shear(object):
    """Shear images.

    Args:
        magnitude (int | float): The magnitude used for shear.
        pad_val (int, tuple[int]): Pixel pad_val value for constant fill. If a
            tuple of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing Shear therefore should be
            in range [0, 1]. Defaults to 0.5.
        direction (str): The shearing direction. Options are 'horizontal' and
            'vertical'. Defaults to 'horizontal'.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'bicubic'.
    """

    def __init__(self,
                 magnitude,
                 pad_val=128,
                 prob=0.5,
                 direction='horizontal',
                 random_negative_prob=0.5,
                 interpolation='bicubic'):
        assert isinstance(magnitude, (int, float)), 'The magnitude type must '\
            f'be int or float, but got {type(magnitude)} instead.'
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, tuple):
            assert len(pad_val) == 3, 'pad_val as a tuple must have 3 ' \
                f'elements, got {len(pad_val)} instead.'
            assert all(isinstance(i, int) for i in pad_val), 'pad_val as a '\
                'tuple must got elements of int type.'
        else:
            raise TypeError('pad_val must be int or tuple with 3 elements.')
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'
        assert direction in ('horizontal', 'vertical'), 'direction must be ' \
            f'either "horizontal" or "vertical", got {direction} instead.'
        assert 0 <= random_negative_prob <= 1.0, 'The random_negative_prob ' \
            f'should be in range [0,1], got {random_negative_prob} instead.'

        self.magnitude = magnitude
        self.pad_val = pad_val
        self.prob = prob
        self.direction = direction
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_sheared = mmcv.imshear(
                img,
                magnitude,
                direction=self.direction,
                border_value=self.pad_val,
                interpolation=self.interpolation)
            results[key] = img_sheared.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'direction={self.direction}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@PIPELINES.register_module()
class Translate(object):
    """Translate images.

    Args:
        magnitude (int | float): The magnitude used for translate. Note that
            the offset is calculated by magnitude * size in the corresponding
            direction. With a magnitude of 1, the whole image will be moved out
             of the range.
        pad_val (int, tuple[int]): Pixel pad_val value for constant fill. If a
            tuple of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing translate therefore should
             be in range [0, 1]. Defaults to 0.5.
        direction (str): The translating direction. Options are 'horizontal'
            and 'vertical'. Defaults to 'horizontal'.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'nearest'.
    """

    def __init__(self,
                 magnitude,
                 pad_val=128,
                 prob=0.5,
                 direction='horizontal',
                 random_negative_prob=0.5,
                 interpolation='nearest'):
        assert isinstance(magnitude, (int, float)), 'The magnitude type must '\
            f'be int or float, but got {type(magnitude)} instead.'
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, tuple):
            assert len(pad_val) == 3, 'pad_val as a tuple must have 3 ' \
                f'elements, got {len(pad_val)} instead.'
            assert all(isinstance(i, int) for i in pad_val), 'pad_val as a '\
                'tuple must got elements of int type.'
        else:
            raise TypeError('pad_val must be int or tuple with 3 elements.')
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'
        assert direction in ('horizontal', 'vertical'), 'direction must be ' \
            f'either "horizontal" or "vertical", got {direction} instead.'
        assert 0 <= random_negative_prob <= 1.0, 'The random_negative_prob ' \
            f'should be in range [0,1], got {random_negative_prob} instead.'

        self.magnitude = magnitude
        self.pad_val = pad_val
        self.prob = prob
        self.direction = direction
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            height, width = img.shape[:2]
            if self.direction == 'horizontal':
                offset = magnitude * width
            else:
                offset = magnitude * height
            img_translated = mmcv.imtranslate(
                img,
                offset,
                direction=self.direction,
                border_value=self.pad_val,
                interpolation=self.interpolation)
            results[key] = img_translated.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'direction={self.direction}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@PIPELINES.register_module()
class Rotate(object):
    """Rotate images.

    Args:
        angle (float): The angle used for rotate. Positive values stand for
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
             the source image. If None, the center of the image will be used.
            defaults to None.
        scale (float): Isotropic scale factor. Defaults to 1.0.
        pad_val (int, tuple[int]): Pixel pad_val value for constant fill. If a
            tuple of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing Rotate therefore should be
            in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the angle
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'nearest'.
    """

    def __init__(self,
                 angle,
                 center=None,
                 scale=1.0,
                 pad_val=128,
                 prob=0.5,
                 random_negative_prob=0.5,
                 interpolation='nearest'):
        assert isinstance(angle, float), 'The angle type must be float, but ' \
            f'got {type(angle)} instead.'
        if isinstance(center, tuple):
            assert len(center) == 2, 'center as a tuple must have 2 ' \
                f'elements, got {len(center)} elements instead.'
        else:
            assert center is None, 'The center type' \
                f'must be tuple or None, got {type(center)} instead.'
        assert isinstance(scale, float), 'the scale type must be float, but ' \
            f'got {type(scale)} instead.'
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, tuple):
            assert len(pad_val) == 3, 'pad_val as a tuple must have 3 ' \
                f'elements, got {len(pad_val)} instead.'
            assert all(isinstance(i, int) for i in pad_val), 'pad_val as a '\
                'tuple must got elements of int type.'
        else:
            raise TypeError('pad_val must be int or tuple with 3 elements.')
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'
        assert 0 <= random_negative_prob <= 1.0, 'The random_negative_prob ' \
            f'should be in range [0,1], got {random_negative_prob} instead.'

        self.angle = angle
        self.center = center
        self.scale = scale
        self.pad_val = pad_val
        self.prob = prob
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        angle = random_negative(self.angle, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_rotated = mmcv.imrotate(
                img,
                angle,
                center=self.center,
                scale=self.scale,
                border_value=self.pad_val,
                interpolation=self.interpolation)
            results[key] = img_rotated.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(angle={self.angle}, '
        repr_str += f'center={self.center}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@PIPELINES.register_module()
class AutoContrast(object):
    """Auto adjust image contrast.

    Args:
        prob (float): The probability for performing invert therefore should
             be in range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'

        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_contrasted = mmcv.auto_contrast(img)
            results[key] = img_contrasted.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class Invert(object):
    """Invert images.

    Args:
        prob (float): The probability for performing invert therefore should
             be in range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'

        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_inverted = mmcv.iminvert(img)
            results[key] = img_inverted.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class Equalize(object):
    """Equalize the image histogram.

    Args:
        prob (float): The probability for performing invert therefore should
             be in range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'

        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_equalized = mmcv.imequalize(img)
            results[key] = img_equalized.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class Solarize(object):
    """Solarize images (invert all pixel values above a threshold).

    Args:
        thr (int | float): The threshold above which the pixels value will be
            inverted.
        prob (float): The probability for solarizing therefore should be in
            range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, thr, prob=0.5):
        assert isinstance(thr, (int, float)), 'The thr type must '\
            f'be int or float, but got {type(thr)} instead.'
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'

        self.thr = thr
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_solarized = mmcv.solarize(img, thr=self.thr)
            results[key] = img_solarized.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(thr={self.thr}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class Posterize(object):
    """Posterize images (reduce the number of bits for each color channel).

    Args:
        bits (int): Number of bits for each pixel in the output img, which
            should be less or equal to 8.
        prob (float): The probability for posterizing therefore should be in
            range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, bits, prob=0.5):
        assert isinstance(bits, int), 'The bits type must be int, '\
            f'but got {type(bits)} instead.'
        assert bits <= 8, f'The bits must be less than 8, got {bits} instead.'
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'

        self.bits = bits
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_posterized = mmcv.posterize(img, bits=self.bits)
            results[key] = img_posterized.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(bits={self.bits}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class Contrast(object):
    """Adjust images contrast.

    Args:
        magnitude (int | float): The magnitude used for adjusting contrast. A
            positive magnitude would enhance the contrast and a negative
            magnitude would make the image grayer. A magnitude=0 gives the
            origin img.
        prob (float): The probability for performing contrast adjusting
            therefore should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
    """

    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float)), 'The magnitude type must '\
            f'be int or float, but got {type(magnitude)} instead.'
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'
        assert 0 <= random_negative_prob <= 1.0, 'The random_negative_prob ' \
            f'should be in range [0,1], got {random_negative_prob} instead.'

        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_contrasted = mmcv.adjust_contrast(img, factor=1 + magnitude)
            results[key] = img_contrasted.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob})'
        return repr_str


@PIPELINES.register_module()
class ColorTransform(object):
    """Adjust images color balance.

    Args:
        magnitude (int | float): The magnitude used for color transform. A
            positive magnitude would enhance the color and a negative magnitude
             would make the image grayer. A magnitude=0 gives the origin img.
        prob (float): The probability for performing ColorTransform therefore
            should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
    """

    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float)), 'The magnitude type must '\
            f'be int or float, but got {type(magnitude)} instead.'
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'
        assert 0 <= random_negative_prob <= 1.0, 'The random_negative_prob ' \
            f'should be in range [0,1], got {random_negative_prob} instead.'

        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_color_adjusted = mmcv.adjust_color(img, alpha=1 + magnitude)
            results[key] = img_color_adjusted.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob})'
        return repr_str


@PIPELINES.register_module()
class Brightness(object):
    """Adjust images brightness.

    Args:
        magnitude (int | float): The magnitude used for adjusting brightness. A
            positive magnitude would enhance the brightness and a negative
            magnitude would make the image darker. A magnitude=0 gives the
            origin img.
        prob (float): The probability for performing contrast adjusting
            therefore should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
    """

    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float)), 'The magnitude type must '\
            f'be int or float, but got {type(magnitude)} instead.'
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'
        assert 0 <= random_negative_prob <= 1.0, 'The random_negative_prob ' \
            f'should be in range [0,1], got {random_negative_prob} instead.'

        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_brightened = mmcv.adjust_brightness(img, factor=1 + magnitude)
            results[key] = img_brightened.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob})'
        return repr_str


@PIPELINES.register_module()
class Sharpness(object):
    """Adjust images sharpness.

    Args:
        magnitude (int | float): The magnitude used for adjusting sharpness. A
            positive magnitude would enhance the sharpness and a negative
            magnitude would make the image bulr. A magnitude=0 gives the
            origin img.
        prob (float): The probability for performing contrast adjusting
            therefore should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
    """

    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float)), 'The magnitude type must '\
            f'be int or float, but got {type(magnitude)} instead.'
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'
        assert 0 <= random_negative_prob <= 1.0, 'The random_negative_prob ' \
            f'should be in range [0,1], got {random_negative_prob} instead.'

        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_sharpened = mmcv.adjust_sharpness(img, factor=1 + magnitude)
            results[key] = img_sharpened.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob})'
        return repr_str

# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import random
from math import ceil
from numbers import Number
from typing import Sequence

import mmcv
import numpy as np

from ..builder import PIPELINES
from .compose import Compose

# Default hyperparameters for all Ops
_HPARAMS_DEFAULT = dict(pad_val=128)


def random_negative(value, random_negative_prob):
    """Randomly negate value based on random_negative_prob."""
    return -value if np.random.rand() < random_negative_prob else value


def merge_hparams(policy: dict, hparams: dict):
    """Merge hyperparameters into policy config.

    Only merge partial hyperparameters required of the policy.

    Args:
        policy (dict): Original policy config dict.
        hparams (dict): Hyperparameters need to be merged.

    Returns:
        dict: Policy config dict after adding ``hparams``.
    """
    op = PIPELINES.get(policy['type'])
    assert op is not None, f'Invalid policy type "{policy["type"]}".'
    for key, value in hparams.items():
        if policy.get(key, None) is not None:
            continue
        if key in inspect.getfullargspec(op.__init__).args:
            policy[key] = value
    return policy


@PIPELINES.register_module()
class AutoAugment(object):
    """Auto augmentation.

    This data augmentation is proposed in `AutoAugment: Learning Augmentation
    Policies from Data <https://arxiv.org/abs/1805.09501>`_.

    Args:
        policies (list[list[dict]]): The policies of auto augmentation. Each
            policy in ``policies`` is a specific augmentation policy, and is
            composed by several augmentations (dict). When AutoAugment is
            called, a random policy in ``policies`` will be selected to
            augment images.
        hparams (dict): Configs of hyperparameters. Hyperparameters will be
            used in policies that require these arguments if these arguments
            are not set in policy dicts. Defaults to use _HPARAMS_DEFAULT.
    """

    def __init__(self, policies, hparams=_HPARAMS_DEFAULT):
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0, \
                'Each policy in policies must be a non-empty list.'
            for augment in policy:
                assert isinstance(augment, dict) and 'type' in augment, \
                    'Each specific augmentation must be a dict with key' \
                    ' "type".'

        self.hparams = hparams
        policies = copy.deepcopy(policies)
        self.policies = []
        for sub in policies:
            merged_sub = [merge_hparams(policy, hparams) for policy in sub]
            self.policies.append(merged_sub)

        self.sub_policy = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        sub_policy = random.choice(self.sub_policy)
        return sub_policy(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(policies={self.policies})'
        return repr_str


@PIPELINES.register_module()
class RandAugment(object):
    r"""Random augmentation.

    This data augmentation is proposed in `RandAugment: Practical automated
    data augmentation with a reduced search space
    <https://arxiv.org/abs/1909.13719>`_.

    Args:
        policies (list[dict]): The policies of random augmentation. Each
            policy in ``policies`` is one specific augmentation policy (dict).
            The policy shall at least have key `type`, indicating the type of
            augmentation. For those which have magnitude, (given to the fact
            they are named differently in different augmentation, )
            `magnitude_key` and `magnitude_range` shall be the magnitude
            argument (str) and the range of magnitude (tuple in the format of
            (val1, val2)), respectively. Note that val1 is not necessarily
            less than val2.
        num_policies (int): Number of policies to select from policies each
            time.
        magnitude_level (int | float): Magnitude level for all the augmentation
            selected.
        total_level (int | float): Total level for the magnitude. Defaults to
            30.
        magnitude_std (Number | str): Deviation of magnitude noise applied.

            - If positive number, magnitude is sampled from normal distribution
              (mean=magnitude, std=magnitude_std).
            - If 0 or negative number, magnitude remains unchanged.
            - If str "inf", magnitude is sampled from uniform distribution
              (range=[min, magnitude]).
        hparams (dict): Configs of hyperparameters. Hyperparameters will be
            used in policies that require these arguments if these arguments
            are not set in policy dicts. Defaults to use _HPARAMS_DEFAULT.

    Note:
        `magnitude_std` will introduce some randomness to policy, modified by
        https://github.com/rwightman/pytorch-image-models.

        When magnitude_std=0, we calculate the magnitude as follows:

        .. math::
            \text{magnitude} = \frac{\text{magnitude_level}}
            {\text{totallevel}} \times (\text{val2} - \text{val1})
            + \text{val1}
    """

    def __init__(self,
                 policies,
                 num_policies,
                 magnitude_level,
                 magnitude_std=0.,
                 total_level=30,
                 hparams=_HPARAMS_DEFAULT):
        assert isinstance(num_policies, int), 'Number of policies must be ' \
            f'of int type, got {type(num_policies)} instead.'
        assert isinstance(magnitude_level, (int, float)), \
            'Magnitude level must be of int or float type, ' \
            f'got {type(magnitude_level)} instead.'
        assert isinstance(total_level, (int, float)),  'Total level must be ' \
            f'of int or float type, got {type(total_level)} instead.'
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'

        assert isinstance(magnitude_std, (Number, str)), \
            'Magnitude std must be of number or str type, ' \
            f'got {type(magnitude_std)} instead.'
        if isinstance(magnitude_std, str):
            assert magnitude_std == 'inf', \
                'Magnitude std must be of number or "inf", ' \
                f'got "{magnitude_std}" instead.'

        assert num_policies > 0, 'num_policies must be greater than 0.'
        assert magnitude_level >= 0, 'magnitude_level must be no less than 0.'
        assert total_level > 0, 'total_level must be greater than 0.'

        self.num_policies = num_policies
        self.magnitude_level = magnitude_level
        self.magnitude_std = magnitude_std
        self.total_level = total_level
        self.hparams = hparams
        policies = copy.deepcopy(policies)
        self._check_policies(policies)
        self.policies = [merge_hparams(policy, hparams) for policy in policies]

    def _check_policies(self, policies):
        for policy in policies:
            assert isinstance(policy, dict) and 'type' in policy, \
                'Each policy must be a dict with key "type".'
            type_name = policy['type']

            magnitude_key = policy.get('magnitude_key', None)
            if magnitude_key is not None:
                assert 'magnitude_range' in policy, \
                    f'RandAugment policy {type_name} needs `magnitude_range`.'
                magnitude_range = policy['magnitude_range']
                assert (isinstance(magnitude_range, Sequence)
                        and len(magnitude_range) == 2), \
                    f'`magnitude_range` of RandAugment policy {type_name} ' \
                    f'should be a Sequence with two numbers.'

    def _process_policies(self, policies):
        processed_policies = []
        for policy in policies:
            processed_policy = copy.deepcopy(policy)
            magnitude_key = processed_policy.pop('magnitude_key', None)
            if magnitude_key is not None:
                magnitude = self.magnitude_level
                # if magnitude_std is positive number or 'inf', move
                # magnitude_value randomly.
                if self.magnitude_std == 'inf':
                    magnitude = random.uniform(0, magnitude)
                elif self.magnitude_std > 0:
                    magnitude = random.gauss(magnitude, self.magnitude_std)
                    magnitude = min(self.total_level, max(0, magnitude))

                val1, val2 = processed_policy.pop('magnitude_range')
                magnitude = (magnitude / self.total_level) * (val2 -
                                                              val1) + val1

                processed_policy.update({magnitude_key: magnitude})
            processed_policies.append(processed_policy)
        return processed_policies

    def __call__(self, results):
        if self.num_policies == 0:
            return results
        sub_policy = random.choices(self.policies, k=self.num_policies)
        sub_policy = self._process_policies(sub_policy)
        sub_policy = Compose(sub_policy)
        return sub_policy(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(policies={self.policies}, '
        repr_str += f'num_policies={self.num_policies}, '
        repr_str += f'magnitude_level={self.magnitude_level}, '
        repr_str += f'total_level={self.total_level})'
        return repr_str


@PIPELINES.register_module()
class Shear(object):
    """Shear images.

    Args:
        magnitude (int | float): The magnitude used for shear.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
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
        elif isinstance(pad_val, Sequence):
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
        self.pad_val = tuple(pad_val)
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
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
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
        elif isinstance(pad_val, Sequence):
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
        self.pad_val = tuple(pad_val)
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
            Defaults to None.
        scale (float): Isotropic scale factor. Defaults to 1.0.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
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
        elif isinstance(pad_val, Sequence):
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
        self.pad_val = tuple(pad_val)
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
class SolarizeAdd(object):
    """SolarizeAdd images (add a certain value to pixels below a threshold).

    Args:
        magnitude (int | float): The value to be added to pixels below the thr.
        thr (int | float): The threshold below which the pixels value will be
            adjusted.
        prob (float): The probability for solarizing therefore should be in
            range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, magnitude, thr=128, prob=0.5):
        assert isinstance(magnitude, (int, float)), 'The thr magnitude must '\
            f'be int or float, but got {type(magnitude)} instead.'
        assert isinstance(thr, (int, float)), 'The thr type must '\
            f'be int or float, but got {type(thr)} instead.'
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'

        self.magnitude = magnitude
        self.thr = thr
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_solarized = np.where(img < self.thr,
                                     np.minimum(img + self.magnitude, 255),
                                     img)
            results[key] = img_solarized.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'thr={self.thr}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class Posterize(object):
    """Posterize images (reduce the number of bits for each color channel).

    Args:
        bits (int | float): Number of bits for each pixel in the output img,
            which should be less or equal to 8.
        prob (float): The probability for posterizing therefore should be in
            range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, bits, prob=0.5):
        assert bits <= 8, f'The bits must be less than 8, got {bits} instead.'
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'

        # To align timm version, we need to round up to integer here.
        self.bits = ceil(bits)
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


@PIPELINES.register_module()
class Cutout(object):
    """Cutout images.

    Args:
        shape (int | float | tuple(int | float)): Expected cutout shape (h, w).
            If given as a single value, the value will be used for
            both h and w.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If it is a sequence, it must have the same length with the image
            channels. Defaults to 128.
        prob (float): The probability for performing cutout therefore should
            be in range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, shape, pad_val=128, prob=0.5):
        if isinstance(shape, float):
            shape = int(shape)
        elif isinstance(shape, tuple):
            shape = tuple(int(i) for i in shape)
        elif not isinstance(shape, int):
            raise TypeError(
                'shape must be of '
                f'type int, float or tuple, got {type(shape)} instead')
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, Sequence):
            assert len(pad_val) == 3, 'pad_val as a tuple must have 3 ' \
                f'elements, got {len(pad_val)} instead.'
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'

        self.shape = shape
        self.pad_val = tuple(pad_val)
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_cutout = mmcv.cutout(img, self.shape, pad_val=self.pad_val)
            results[key] = img_cutout.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(shape={self.shape}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str

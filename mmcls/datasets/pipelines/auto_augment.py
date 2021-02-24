import copy

import mmcv
import numpy as np

from ..builder import PIPELINES
from .compose import Compose

try:
    from PIL import Image
except ImportError:
    Image = None

cv2_allowed_interp = ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']

if Image is not None:
    pillow_interp_codes = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'box': Image.BOX,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }
    pillow_allowed_interp = [
        'nearest', 'bilinear', 'bicubic', 'box', 'lanczos', 'hamming'
    ]


def random_negative(value, random_negative_prob):
    """Randomly negate value based on random_negative_prob."""
    return -value if np.random.rand() < random_negative_prob else value


@PIPELINES.register_module()
class AutoAugment(object):
    """Auto Augmentation.

    This data augmentation is proposed in `AutoAugment: Learning Augmentation
    Strategies From Data <https://ieeexplore.ieee.org/document/8953317>`_.

    Args:
        policies (list[list[dict]]| str): The policies of auto augmentation.
            When policies is list, each policy in ``policies`` is a specific
            augmentation policy, and is composed by several augmentations
            (dict). When AutoAugment is called, a random policy in ``policies``
             will be selected to augment images. When policies is str, the
            predetermined policies will be used.
    """

    def __init__(self, policies):
        if isinstance(policies, str):
            if policies == 'ImageNetPolicy':
                policies = []  # to be fixed
            elif policies == 'CifarPolicy':
                policies = []  # to be fixed
            else:
                raise ValueError(f'Policies: {policies} is not supported for '
                                 'auto augmentation. Supported policies are '
                                 '"ImageNetPolicy", "CifarPolicy".')
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
        self.transforms = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        transform = np.random.choice(self.transforms)
        return transform(results)

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.policies})'


PIPELINES.register_module()


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
            'vertical'. Default: 'horizontal'.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
            Defaults to 'bilinear'.
        backend (str): The backend type used for shear. Options are
            cv2 and pillow. Defaults to pillow.
    """

    def __init__(self,
                 magnitude,
                 pad_val=128,
                 prob=0.5,
                 direction='horizontal',
                 random_negative_prob=0.5,
                 interpolation='bilinear',
                 backend='pillow'):
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
        if backend == 'cv2':
            assert interpolation in cv2_allowed_interp
        elif backend == 'pillow':
            assert interpolation in pillow_allowed_interp
        else:
            raise ValueError(f'backend: {backend} is not supported for shear.'
                             'Supported backends are "cv2", "pillow".')

        self.magnitude = magnitude
        self.pad_val = pad_val
        self.prob = prob
        self.direction = direction
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation
        self.backend = backend

    def __call__(self, results):
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            if self.backend == 'cv2':
                results[key] = mmcv.imshear(
                    results[key],
                    magnitude,
                    border_value=self.pad_val,
                    interpolation=self.interpolation)
            else:
                assert results[key].dtype == np.uint8, \
                    'Pillow backend only support uint8 type'
                interpolation = pillow_interp_codes[self.interpolation]
                img_pil = Image.fromarray(results[key])
                img_pil = img_pil.transform(
                    img_pil.size,
                    Image.AFFINE, (1, magnitude, 0, 0, 1, 0),
                    interpolation,
                    fillcolor=self.pad_val)
                results[key] = np.array(img_pil)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'(pad_val={self.pad_val}, '
        repr_str += f'(prob={self.prob}, '
        repr_str += f'(direction={self.direction}, '
        repr_str += f'(random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation}, '
        repr_str += f'backend={self.backend}, '
        return repr_str

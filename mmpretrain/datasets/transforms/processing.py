# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import math
import numbers
from numbers import Number
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import mmengine
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmpretrain.registry import TRANSFORMS

try:
    import albumentations
except ImportError:
    albumentations = None


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Crop the given Image at a random location.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        crop_size (int | Sequence): Desired output size of the crop. If
            crop_size is an int instead of sequence like (h, w), a square crop
            (crop_size, crop_size) is made.
        padding (int | Sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to
            pad left, top, right, bottom borders respectively.  If a sequence
            of length 2 is provided, it is used to pad left/right, top/bottom
            borders, respectively. Default: None, which means no padding.
        pad_if_needed (bool): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
            Default: False.
        pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
            fill. If a tuple of length 3, it is used to pad_val R, G, B
            channels respectively. Default: 0.
        padding_mode (str): Type of padding. Defaults to "constant". Should
            be one of the following:

            - ``constant``: Pads with a constant value, this value is specified
              with pad_val.
            - ``edge``: pads with the last value at the edge of the image.
            - ``reflect``: Pads with reflection of image without repeating the
              last value on the edge. For example, padding [1, 2, 3, 4]
              with 2 elements on both sides in reflect mode will result
              in [3, 2, 1, 2, 3, 4, 3, 2].
            - ``symmetric``: Pads with reflection of image repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with
              2 elements on both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3].
    """

    def __init__(self,
                 crop_size: Union[Sequence, int],
                 padding: Optional[Union[Sequence, int]] = None,
                 pad_if_needed: bool = False,
                 pad_val: Union[Number, Sequence[Number]] = 0,
                 padding_mode: str = 'constant'):
        if isinstance(crop_size, Sequence):
            assert len(crop_size) == 2
            assert crop_size[0] > 0 and crop_size[1] > 0
            self.crop_size = crop_size
        else:
            assert crop_size > 0
            self.crop_size = (crop_size, crop_size)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        target_h, target_w = self.crop_size
        if w == target_w and h == target_h:
            return 0, 0, h, w
        elif w < target_w or h < target_h:
            target_w = min(w, target_w)
            target_h = min(w, target_h)

        offset_h = np.random.randint(0, h - target_h + 1)
        offset_w = np.random.randint(0, w - target_w + 1)

        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        if self.padding is not None:
            img = mmcv.impad(img, padding=self.padding, pad_val=self.pad_val)

        # pad img if needed
        if self.pad_if_needed:
            h_pad = math.ceil(max(0, self.crop_size[0] - img.shape[0]) / 2)
            w_pad = math.ceil(max(0, self.crop_size[1] - img.shape[1]) / 2)

            img = mmcv.impad(
                img,
                padding=(w_pad, h_pad, w_pad, h_pad),
                pad_val=self.pad_val,
                padding_mode=self.padding_mode)

        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        img = mmcv.imcrop(
            img,
            np.array([
                offset_w,
                offset_h,
                offset_w + target_w - 1,
                offset_h + target_h - 1,
            ]))
        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        repr_str += f', padding={self.padding}'
        repr_str += f', pad_if_needed={self.pad_if_needed}'
        repr_str += f', pad_val={self.pad_val}'
        repr_str += f', padding_mode={self.padding_mode})'
        return repr_str


@TRANSFORMS.register_module()
class RandomResizedCrop(BaseTransform):
    """Crop the given image to random scale and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        scale (sequence | int): Desired output scale of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        crop_ratio_range (tuple): Range of the random size of the cropped
            image compared to the original image. Defaults to (0.08, 1.0).
        aspect_ratio_range (tuple): Range of the random aspect ratio of the
            cropped image compared to the original image.
            Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            'cv2' and 'pillow'. Defaults to 'cv2'.
    """

    def __init__(self,
                 scale: Union[Sequence, int],
                 crop_ratio_range: Tuple[float, float] = (0.08, 1.0),
                 aspect_ratio_range: Tuple[float, float] = (3. / 4., 4. / 3.),
                 max_attempts: int = 10,
                 interpolation: str = 'bilinear',
                 backend: str = 'cv2') -> None:
        if isinstance(scale, Sequence):
            assert len(scale) == 2
            assert scale[0] > 0 and scale[1] > 0
            self.scale = scale
        else:
            assert scale > 0
            self.scale = (scale, scale)
        if (crop_ratio_range[0] > crop_ratio_range[1]) or (
                aspect_ratio_range[0] > aspect_ratio_range[1]):
            raise ValueError(
                'range should be of kind (min, max). '
                f'But received crop_ratio_range {crop_ratio_range} '
                f'and aspect_ratio_range {aspect_ratio_range}.')
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')

        self.crop_ratio_range = crop_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.backend = backend

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w

        for _ in range(self.max_attempts):
            target_area = np.random.uniform(*self.crop_ratio_range) * area
            log_ratio = (math.log(self.aspect_ratio_range[0]),
                         math.log(self.aspect_ratio_range[1]))
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))
            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_w <= w and 0 < target_h <= h:
                offset_h = np.random.randint(0, h - target_h + 1)
                offset_w = np.random.randint(0, w - target_w + 1)

                return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.aspect_ratio_range):
            target_w = w
            target_h = int(round(target_w / min(self.aspect_ratio_range)))
        elif in_ratio > max(self.aspect_ratio_range):
            target_h = h
            target_w = int(round(target_h * max(self.aspect_ratio_range)))
        else:  # whole image
            target_w = w
            target_h = h
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly resized crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly resized cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        img = mmcv.imcrop(
            img,
            bboxes=np.array([
                offset_w, offset_h, offset_w + target_w - 1,
                offset_h + target_h - 1
            ]))
        img = mmcv.imresize(
            img,
            tuple(self.scale[::-1]),
            interpolation=self.interpolation,
            backend=self.backend)
        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(scale={self.scale}'
        repr_str += ', crop_ratio_range='
        repr_str += f'{tuple(round(s, 4) for s in self.crop_ratio_range)}'
        repr_str += ', aspect_ratio_range='
        repr_str += f'{tuple(round(r, 4) for r in self.aspect_ratio_range)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


@TRANSFORMS.register_module()
class EfficientNetRandomCrop(RandomResizedCrop):
    """EfficientNet style RandomResizedCrop.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        scale (int): Desired output scale of the crop. Only int size is
            accepted, a square crop (size, size) is made.
        min_covered (Number): Minimum ratio of the cropped area to the original
             area. Defaults to 0.1.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Defaults to 32.
        crop_ratio_range (tuple): Range of the random size of the cropped
            image compared to the original image. Defaults to (0.08, 1.0).
        aspect_ratio_range (tuple): Range of the random aspect ratio of the
            cropped image compared to the original image.
            Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bicubic'.
        backend (str): The image resize backend type, accepted values are
            'cv2' and 'pillow'. Defaults to 'cv2'.
    """

    def __init__(self,
                 scale: int,
                 min_covered: float = 0.1,
                 crop_padding: int = 32,
                 interpolation: str = 'bicubic',
                 **kwarg):
        assert isinstance(scale, int)
        super().__init__(scale, interpolation=interpolation, **kwarg)
        assert min_covered >= 0, 'min_covered should be no less than 0.'
        assert crop_padding >= 0, 'crop_padding should be no less than 0.'

        self.min_covered = min_covered
        self.crop_padding = crop_padding

    # https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/data.py # noqa
    @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w
        min_target_area = self.crop_ratio_range[0] * area
        max_target_area = self.crop_ratio_range[1] * area

        for _ in range(self.max_attempts):
            aspect_ratio = np.random.uniform(*self.aspect_ratio_range)
            min_target_h = int(
                round(math.sqrt(min_target_area / aspect_ratio)))
            max_target_h = int(
                round(math.sqrt(max_target_area / aspect_ratio)))

            if max_target_h * aspect_ratio > w:
                max_target_h = int((w + 0.5 - 1e-7) / aspect_ratio)
                if max_target_h * aspect_ratio > w:
                    max_target_h -= 1

            max_target_h = min(max_target_h, h)
            min_target_h = min(max_target_h, min_target_h)

            # slightly differs from tf implementation
            target_h = int(
                round(np.random.uniform(min_target_h, max_target_h)))
            target_w = int(round(target_h * aspect_ratio))
            target_area = target_h * target_w

            # slight differs from tf. In tf, if target_area > max_target_area,
            # area will be recalculated
            if (target_area < min_target_area or target_area > max_target_area
                    or target_w > w or target_h > h
                    or target_area < self.min_covered * area):
                continue

            offset_h = np.random.randint(0, h - target_h + 1)
            offset_w = np.random.randint(0, w - target_w + 1)

            return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        img_short = min(h, w)
        crop_size = self.scale[0] / (self.scale[0] +
                                     self.crop_padding) * img_short

        offset_h = max(0, int(round((h - crop_size) / 2.)))
        offset_w = max(0, int(round((w - crop_size) / 2.)))
        return offset_h, offset_w, crop_size, crop_size

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = super().__repr__()[:-1]
        repr_str += f', min_covered={self.min_covered}'
        repr_str += f', crop_padding={self.crop_padding})'
        return repr_str


@TRANSFORMS.register_module()
class RandomErasing(BaseTransform):
    """Randomly selects a rectangle region in an image and erase pixels.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img

    Args:
        erase_prob (float): Probability that image will be randomly erased.
            Default: 0.5
        min_area_ratio (float): Minimum erased area / input image area
            Default: 0.02
        max_area_ratio (float): Maximum erased area / input image area
            Default: 0.4
        aspect_range (sequence | float): Aspect ratio range of erased area.
            if float, it will be converted to (aspect_ratio, 1/aspect_ratio)
            Default: (3/10, 10/3)
        mode (str): Fill method in erased area, can be:

            - const (default): All pixels are assign with the same value.
            - rand: each pixel is assigned with a random value in [0, 255]

        fill_color (sequence | Number): Base color filled in erased area.
            Defaults to (128, 128, 128).
        fill_std (sequence | Number, optional): If set and ``mode`` is 'rand',
            fill erased area with random color from normal distribution
            (mean=fill_color, std=fill_std); If not set, fill erased area with
            random color from uniform distribution (0~255). Defaults to None.

    Note:
        See `Random Erasing Data Augmentation
        <https://arxiv.org/pdf/1708.04896.pdf>`_

        This paper provided 4 modes: RE-R, RE-M, RE-0, RE-255, and use RE-M as
        default. The config of these 4 modes are:

        - RE-R: RandomErasing(mode='rand')
        - RE-M: RandomErasing(mode='const', fill_color=(123.67, 116.3, 103.5))
        - RE-0: RandomErasing(mode='const', fill_color=0)
        - RE-255: RandomErasing(mode='const', fill_color=255)
    """

    def __init__(self,
                 erase_prob=0.5,
                 min_area_ratio=0.02,
                 max_area_ratio=0.4,
                 aspect_range=(3 / 10, 10 / 3),
                 mode='const',
                 fill_color=(128, 128, 128),
                 fill_std=None):
        assert isinstance(erase_prob, float) and 0. <= erase_prob <= 1.
        assert isinstance(min_area_ratio, float) and 0. <= min_area_ratio <= 1.
        assert isinstance(max_area_ratio, float) and 0. <= max_area_ratio <= 1.
        assert min_area_ratio <= max_area_ratio, \
            'min_area_ratio should be smaller than max_area_ratio'
        if isinstance(aspect_range, float):
            aspect_range = min(aspect_range, 1 / aspect_range)
            aspect_range = (aspect_range, 1 / aspect_range)
        assert isinstance(aspect_range, Sequence) and len(aspect_range) == 2 \
            and all(isinstance(x, float) for x in aspect_range), \
            'aspect_range should be a float or Sequence with two float.'
        assert all(x > 0 for x in aspect_range), \
            'aspect_range should be positive.'
        assert aspect_range[0] <= aspect_range[1], \
            'In aspect_range (min, max), min should be smaller than max.'
        assert mode in ['const', 'rand'], \
            'Please select `mode` from ["const", "rand"].'
        if isinstance(fill_color, Number):
            fill_color = [fill_color] * 3
        assert isinstance(fill_color, Sequence) and len(fill_color) == 3 \
            and all(isinstance(x, Number) for x in fill_color), \
            'fill_color should be a float or Sequence with three int.'
        if fill_std is not None:
            if isinstance(fill_std, Number):
                fill_std = [fill_std] * 3
            assert isinstance(fill_std, Sequence) and len(fill_std) == 3 \
                and all(isinstance(x, Number) for x in fill_std), \
                'fill_std should be a float or Sequence with three int.'

        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color
        self.fill_std = fill_std

    def _fill_pixels(self, img, top, left, h, w):
        """Fill pixels to the patch of image."""
        if self.mode == 'const':
            patch = np.empty((h, w, 3), dtype=np.uint8)
            patch[:, :] = np.array(self.fill_color, dtype=np.uint8)
        elif self.fill_std is None:
            # Uniform distribution
            patch = np.random.uniform(0, 256, (h, w, 3)).astype(np.uint8)
        else:
            # Normal distribution
            patch = np.random.normal(self.fill_color, self.fill_std, (h, w, 3))
            patch = np.clip(patch.astype(np.int32), 0, 255).astype(np.uint8)

        img[top:top + h, left:left + w] = patch
        return img

    @cache_randomness
    def random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.erase_prob

    @cache_randomness
    def random_patch(self, img_h, img_w):
        """Randomly generate patch the erase."""
        # convert the aspect ratio to log space to equally handle width and
        # height.
        log_aspect_range = np.log(
            np.array(self.aspect_range, dtype=np.float32))
        aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
        area = img_h * img_w
        area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

        h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
        w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
        top = np.random.randint(0, img_h - h) if img_h > h else 0
        left = np.random.randint(0, img_w - w) if img_w > w else 0
        return top, left, h, w

    def transform(self, results):
        """
        Args:
            results (dict): Results dict from pipeline

        Returns:
            dict: Results after the transformation.
        """
        if self.random_disable():
            return results

        img = results['img']
        img_h, img_w = img.shape[:2]

        img = self._fill_pixels(img, *self.random_patch(img_h, img_w))

        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(erase_prob={self.erase_prob}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_area_ratio={self.max_area_ratio}, '
        repr_str += f'aspect_range={self.aspect_range}, '
        repr_str += f'mode={self.mode}, '
        repr_str += f'fill_color={self.fill_color}, '
        repr_str += f'fill_std={self.fill_std})'
        return repr_str


@TRANSFORMS.register_module()
class EfficientNetCenterCrop(BaseTransform):
    r"""EfficientNet style center crop.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        crop_size (int): Expected size after cropping with the format
            of (h, w).
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Defaults to 32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Only valid if
            ``efficientnet_style`` is True. Defaults to 'bicubic'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Only valid if efficientnet style is True.
            Defaults to `cv2`.
    Notes:
        - If the image is smaller than the crop size, return the original
          image.
        - The pipeline will be to first
          to perform the center crop with the ``crop_size_`` as:

        .. math::

            \text{crop_size_} = \frac{\text{crop_size}}{\text{crop_size} +
            \text{crop_padding}} \times \text{short_edge}

        And then the pipeline resizes the img to the input crop size.
    """

    def __init__(self,
                 crop_size: int,
                 crop_padding: int = 32,
                 interpolation: str = 'bicubic',
                 backend: str = 'cv2'):
        assert isinstance(crop_size, int)
        assert crop_size > 0
        assert crop_padding >= 0
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')

        self.crop_size = crop_size
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    def transform(self, results: dict) -> dict:
        """Transform function to randomly resized crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: EfficientNet style center cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        h, w = img.shape[:2]

        # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L118 # noqa
        img_short = min(h, w)
        crop_size = self.crop_size / (self.crop_size +
                                      self.crop_padding) * img_short

        offset_h = max(0, int(round((h - crop_size) / 2.)))
        offset_w = max(0, int(round((w - crop_size) / 2.)))

        # crop the image
        img = mmcv.imcrop(
            img,
            bboxes=np.array([
                offset_w, offset_h, offset_w + crop_size - 1,
                offset_h + crop_size - 1
            ]))
        # resize image
        img = mmcv.imresize(
            img, (self.crop_size, self.crop_size),
            interpolation=self.interpolation,
            backend=self.backend)
        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        repr_str += f', crop_padding={self.crop_padding}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


@TRANSFORMS.register_module()
class ResizeEdge(BaseTransform):
    """Resize images along the specified edge.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    **Added Keys:**

    - scale
    - scale_factor

    Args:
        scale (int): The edge scale to resizing.
        edge (str): The edge to resize. Defaults to 'short'.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results.
            Defaults to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
            Defaults to 'bilinear'.
    """

    def __init__(self,
                 scale: int,
                 edge: str = 'short',
                 backend: str = 'cv2',
                 interpolation: str = 'bilinear') -> None:
        allow_edges = ['short', 'long', 'width', 'height']
        assert edge in allow_edges, \
            f'Invalid edge "{edge}", please specify from {allow_edges}.'
        self.edge = edge
        self.scale = scale
        self.backend = backend
        self.interpolation = interpolation

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        img, w_scale, h_scale = mmcv.imresize(
            results['img'],
            results['scale'],
            interpolation=self.interpolation,
            return_scale=True,
            backend=self.backend)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['scale'] = img.shape[:2][::-1]
        results['scale_factor'] = (w_scale, h_scale)

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img', 'scale', 'scale_factor',
            'img_shape' keys are updated in result dict.
        """
        assert 'img' in results, 'No `img` field in the input.'

        h, w = results['img'].shape[:2]
        if any([
                # conditions to resize the width
                self.edge == 'short' and w < h,
                self.edge == 'long' and w > h,
                self.edge == 'width',
        ]):
            width = self.scale
            height = int(self.scale * h / w)
        else:
            height = self.scale
            width = int(self.scale * w / h)
        results['scale'] = (width, height)

        self._resize_img(results)
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'edge={self.edge}, '
        repr_str += f'backend={self.backend}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    """Randomly change the brightness, contrast and saturation of an image.

    Modified from
    https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
    Licensed under the BSD 3-Clause License.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img

    Args:
        brightness (float | Sequence[float] (min, max)): How much to jitter
            brightness. brightness_factor is chosen uniformly from
            ``[max(0, 1 - brightness), 1 + brightness]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        contrast (float | Sequence[float] (min, max)): How much to jitter
            contrast. contrast_factor is chosen uniformly from
            ``[max(0, 1 - contrast), 1 + contrast]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        saturation (float | Sequence[float] (min, max)): How much to jitter
            saturation. saturation_factor is chosen uniformly from
            ``[max(0, 1 - saturation), 1 + saturation]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        hue (float | Sequence[float] (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from ``[-hue, hue]`` (0 <= hue
            <= 0.5) or the given ``[min, max]`` (-0.5 <= min <= max <= 0.5).
            Defaults to 0.
    """

    def __init__(self,
                 brightness: Union[float, Sequence[float]] = 0.,
                 contrast: Union[float, Sequence[float]] = 0.,
                 saturation: Union[float, Sequence[float]] = 0.,
                 hue: Union[float, Sequence[float]] = 0.):
        self.brightness = self._set_range(brightness, 'brightness')
        self.contrast = self._set_range(contrast, 'contrast')
        self.saturation = self._set_range(saturation, 'saturation')
        self.hue = self._set_range(hue, 'hue', center=0, bound=(-0.5, 0.5))

    def _set_range(self, value, name, center=1, bound=(0, float('inf'))):
        """Set the range of magnitudes."""
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f'If {name} is a single number, it must be non negative.')
            value = (center - float(value), center + float(value))

        if isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                value = np.clip(value, bound[0], bound[1])
                from mmengine.logging import MMLogger
                logger = MMLogger.get_current_instance()
                logger.warning(f'ColorJitter {name} values exceed the bound '
                               f'{bound}, clipped to the bound.')
        else:
            raise TypeError(f'{name} should be a single number '
                            'or a list/tuple with length 2.')

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        else:
            value = tuple(value)

        return value

    @cache_randomness
    def _rand_params(self):
        """Get random parameters including magnitudes and indices of
        transforms."""
        trans_inds = np.random.permutation(4)
        b, c, s, h = (None, ) * 4

        if self.brightness is not None:
            b = np.random.uniform(self.brightness[0], self.brightness[1])
        if self.contrast is not None:
            c = np.random.uniform(self.contrast[0], self.contrast[1])
        if self.saturation is not None:
            s = np.random.uniform(self.saturation[0], self.saturation[1])
        if self.hue is not None:
            h = np.random.uniform(self.hue[0], self.hue[1])

        return trans_inds, b, c, s, h

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: ColorJitter results, 'img' key is updated in result dict.
        """
        img = results['img']
        trans_inds, brightness, contrast, saturation, hue = self._rand_params()

        for index in trans_inds:
            if index == 0 and brightness is not None:
                img = mmcv.adjust_brightness(img, brightness)
            elif index == 1 and contrast is not None:
                img = mmcv.adjust_contrast(img, contrast)
            elif index == 2 and saturation is not None:
                img = mmcv.adjust_color(img, alpha=saturation)
            elif index == 3 and hue is not None:
                img = mmcv.adjust_hue(img, hue)

        results['img'] = img
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(brightness={self.brightness}, '
        repr_str += f'contrast={self.contrast}, '
        repr_str += f'saturation={self.saturation}, '
        repr_str += f'hue={self.hue})'
        return repr_str


@TRANSFORMS.register_module()
class Lighting(BaseTransform):
    """Adjust images lighting using AlexNet-style PCA jitter.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img

    Args:
        eigval (Sequence[float]): the eigenvalue of the convariance matrix
            of pixel values, respectively.
        eigvec (list[list]): the eigenvector of the convariance matrix of
            pixel values, respectively.
        alphastd (float): The standard deviation for distribution of alpha.
            Defaults to 0.1.
        to_rgb (bool): Whether to convert img to rgb. Defaults to False.
    """

    def __init__(self,
                 eigval: Sequence[float],
                 eigvec: Sequence[float],
                 alphastd: float = 0.1,
                 to_rgb: bool = False):
        assert isinstance(eigval, Sequence), \
            f'eigval must be Sequence, got {type(eigval)} instead.'
        assert isinstance(eigvec, Sequence), \
            f'eigvec must be Sequence, got {type(eigvec)} instead.'
        for vec in eigvec:
            assert isinstance(vec, Sequence) and len(vec) == len(eigvec[0]), \
                'eigvec must contains lists with equal length.'
        assert isinstance(alphastd, float), 'alphastd should be of type ' \
            f'float or int, got {type(alphastd)} instead.'

        self.eigval = np.array(eigval)
        self.eigvec = np.array(eigvec)
        self.alphastd = alphastd
        self.to_rgb = to_rgb

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Lightinged results, 'img' key is updated in result dict.
        """
        assert 'img' in results, 'No `img` field in the input.'

        img = results['img']
        img_lighting = mmcv.adjust_lighting(
            img,
            self.eigval,
            self.eigvec,
            alphastd=self.alphastd,
            to_rgb=self.to_rgb)
        results['img'] = img_lighting.astype(img.dtype)
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(eigval={self.eigval.tolist()}, '
        repr_str += f'eigvec={self.eigvec.tolist()}, '
        repr_str += f'alphastd={self.alphastd}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


# 'Albu' is used in previous versions of mmpretrain, here is for compatibility
# users can use both 'Albumentations' and 'Albu'.
@TRANSFORMS.register_module(['Albumentations', 'Albu'])
class Albumentations(BaseTransform):
    """Wrapper to use augmentation from albumentations library.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Adds custom transformations from albumentations library.
    More details can be found in
    `Albumentations <https://albumentations.readthedocs.io>`_.
    An example of ``transforms`` is as followed:

    .. code-block::

        [
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

    Args:
        transforms (List[Dict]): List of albumentations transform configs.
        keymap (Optional[Dict]): Mapping of mmpretrain to albumentations
            fields, in format {'input key':'albumentation-style key'}.
            Defaults to None.

    Example:
        >>> import mmcv
        >>> from mmpretrain.datasets import Albumentations
        >>> transforms = [
        ...     dict(
        ...         type='ShiftScaleRotate',
        ...         shift_limit=0.0625,
        ...         scale_limit=0.0,
        ...         rotate_limit=0,
        ...         interpolation=1,
        ...         p=0.5),
        ...     dict(
        ...         type='RandomBrightnessContrast',
        ...         brightness_limit=[0.1, 0.3],
        ...         contrast_limit=[0.1, 0.3],
        ...         p=0.2),
        ...     dict(type='ChannelShuffle', p=0.1),
        ...     dict(
        ...         type='OneOf',
        ...         transforms=[
        ...             dict(type='Blur', blur_limit=3, p=1.0),
        ...             dict(type='MedianBlur', blur_limit=3, p=1.0)
        ...         ],
        ...         p=0.1),
        ... ]
        >>> albu = Albumentations(transforms)
        >>> data = {'img': mmcv.imread('./demo/demo.JPEG')}
        >>> data = albu(data)
        >>> print(data['img'].shape)
        (375, 500, 3)
    """

    def __init__(self, transforms: List[Dict], keymap: Optional[Dict] = None):
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')
        else:
            from albumentations import Compose as albu_Compose

        assert isinstance(transforms, list), 'transforms must be a list.'
        if keymap is not None:
            assert isinstance(keymap, dict), 'keymap must be None or a dict. '

        self.transforms = transforms

        self.aug = albu_Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = dict(img='image')
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: Dict):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg, 'each item in ' \
            "transforms must be a dict with keyword 'type'."
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmengine.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.

        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def transform(self, results: Dict) -> Dict:
        """Transform function to perform albumentations transforms.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results, 'img' and 'img_shape' keys are
                updated in result dict.
        """
        assert 'img' in results, 'No `img` field in the input.'

        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        results = self.aug(**results)

        # back to the original format
        results = self.mapper(results, self.keymap_back)
        results['img_shape'] = results['img'].shape[:2]

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={repr(self.transforms)})'
        return repr_str

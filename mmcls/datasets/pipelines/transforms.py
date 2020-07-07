import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class Resize(object):
    """Resize images.

    Args:
        size (int | tuple): Images scales for resizing (h, w).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            More details can be found in `mmcv.image.geometric`.
    """

    def __init__(self, size, interpolation='bilinear'):
        assert isinstance(size, int) or (isinstance(size, tuple)
                                         and len(size) == 2)
        if isinstance(size, int):
            size = (size, size)
        assert size[0] > 0 and size[1] > 0
        assert interpolation in ("nearest", "bilinear", "bicubic", "area",
                                 "lanczos")

        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.interpolation = interpolation

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            img = mmcv.imresize(
                results[key],
                size=(self.width, self.height),
                interpolation=self.interpolation,
                return_scale=False)
            results[key] = img
            results['img_shape'] = img.shape

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@PIPELINES.register_module()
class CenterCrop(object):
    """Center crop the image.

    Args:
        crop_size (int | tuple): Expected size after cropping, (h, w).

    Notes:
        If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int) or (isinstance(crop_size, tuple)
                                              and len(crop_size) == 2)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_height, img_width, _ = img.shape

            y1 = max(0, int(round((img_height - crop_height) / 2.)))
            x1 = max(0, int(round((img_width - crop_width) / 2.)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1

            # crop the image
            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, '
        repr_str += f'std={list(self.std)}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str

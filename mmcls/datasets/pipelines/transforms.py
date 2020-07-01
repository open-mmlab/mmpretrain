import mmcv
import numpy as np
from torchvision import transforms

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomCrop(transforms.RandomCrop):
    """
    """

    def __init__(self, *args, **kwargs):
        super(RandomCrop, self).__init__(*args, **kwargs)

    def __call__(self, results):
        results['img'] = super(RandomCrop, self).__call__(results['img'])
        return results


@PIPELINES.register_module()
class RandomResizedCrop(transforms.RandomResizedCrop):
    """
    """

    def __init__(self, *args, **kwargs):
        super(RandomResizedCrop, self).__init__(*args, **kwargs)

    def __call__(self, results):
        results['img'] = super(RandomResizedCrop,
                               self).__call__(results['img'])
        return results


@PIPELINES.register_module()
class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """
    """

    def __init__(self, *args, **kwargs):
        super(RandomHorizontalFlip, self).__init__(*args, **kwargs)

    def __call__(self, results):
        results['img'] = super(RandomHorizontalFlip,
                               self).__call__(results['img'])
        return results


@PIPELINES.register_module()
class Resize(transforms.Resize):
    """
    """

    def __init__(self, *args, **kwargs):
        super(Resize, self).__init__(*args, **kwargs)

    def __call__(self, results):
        results['img'] = super(Resize, self).__call__(results['img'])
        return results


@PIPELINES.register_module()
class CenterCrop(transforms.CenterCrop):
    """
    """

    def __init__(self, *args, **kwargs):
        super(CenterCrop, self).__init__(*args, **kwargs)

    def __call__(self, results):
        results['img'] = super(CenterCrop, self).__call__(results['img'])
        return results


@PIPELINES.register_module()
class ColorJitter(transforms.ColorJitter):
    """
    """

    def __init__(self, *args, **kwargs):
        super(ColorJitter, self).__init__(*args, **kwargs)

    def __call__(self, results):
        results['img'] = super(ColorJitter, self).__call__(results['img'])
        return results


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
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

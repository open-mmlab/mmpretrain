# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections.abc import Sequence

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.utils import is_str
from PIL import Image

from mmcls.registry import TRANSFORMS
from mmcls.structures import ClsDataSample


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@TRANSFORMS.register_module()
class PackClsInputs(BaseTransform):
    """Pack the inputs data for the classification.

    **Required Keys:**

    - img
    - gt_label (optional)
    - ``*meta_keys`` (optional)

    **Deleted Keys:**

    All keys in the dict.

    **Added Keys:**

    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmcls.structures.ClsDataSample`): The annotation
      info of the sample.

    Args:
        meta_keys (Sequence[str]): The meta keys to be saved in the
            ``metainfo`` of the packed ``data_samples``.
            Defaults to a tuple includes keys:

            - ``sample_idx``: The id of the image sample.
            - ``img_path``: The path to the image file.
            - ``ori_shape``: The original shape of the image as a tuple (H, W).
            - ``img_shape``: The shape of the image after the pipeline as a
              tuple (H, W).
            - ``scale_factor``: The scale factor between the resized image and
              the original image.
            - ``flip``: A boolean indicating if image flip transform was used.
            - ``flip_direction``: The flipping direction.
    """

    def __init__(self,
                 meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)
        else:
            warnings.warn(
                'Cannot get "img" in the input dict of `PackClsInputs`,'
                'please make sure `LoadImageFromFile` has been added '
                'in the data pipeline or images have been loaded in '
                'the dataset.')

        data_sample = ClsDataSample()
        if 'gt_label' in results:
            gt_label = results['gt_label']
            data_sample.set_gt_label(gt_label)

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class Transpose(BaseTransform):
    """Transpose numpy array.

    **Required Keys:**

    - ``*keys``

    **Modified Keys:**

    - ``*keys``

    Args:
        keys (List[str]): The fields to convert to tensor.
        order (List[int]): The output dimensions order.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def transform(self, results):
        """Method to transpose array."""
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'


@TRANSFORMS.register_module()
class ToPIL(BaseTransform):
    """Convert the image from OpenCV format to :obj:`PIL.Image.Image`.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    """

    def transform(self, results):
        """Method to convert images to :obj:`PIL.Image.Image`."""
        results['img'] = Image.fromarray(results['img'])
        return results


@TRANSFORMS.register_module()
class ToNumpy(BaseTransform):
    """Convert object to :obj:`numpy.ndarray`.

    **Required Keys:**

    - ``*keys**``

    **Modified Keys:**

    - ``*keys**``

    Args:
        dtype (str, optional): The dtype of the converted numpy array.
            Defaults to None.
    """

    def __init__(self, keys, dtype=None):
        self.keys = keys
        self.dtype = dtype

    def transform(self, results):
        """Method to convert object to :obj:`numpy.ndarray`."""
        for key in self.keys:
            results[key] = np.array(results[key], dtype=self.dtype)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, dtype={self.dtype})'


@TRANSFORMS.register_module()
class Collect(BaseTransform):
    """Collect and only reserve the specified fields.

    **Required Keys:**

    - ``*keys``

    **Deleted Keys:**

    All keys except those in the argument ``*keys``.

    Args:
        keys (Sequence[str]): The keys of the fields to be collected.
    """

    def __init__(self, keys):
        self.keys = keys

    def transform(self, results):
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

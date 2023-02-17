# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from collections.abc import Sequence
from functools import partial

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.utils import is_str
from PIL import Image

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import ClsDataSample, MultiTaskDataSample


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
    - data_samples (:obj:`~mmpretrain.structures.ClsDataSample`): The
      annotation info of the sample.

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
class PackMultiTaskInputs(BaseTransform):
    """Convert all image labels of multi-task dataset to a dict of tensor.

    Args:
        tasks (List[str]): The task names defined in the dataset.
        meta_keys(Sequence[str]): The meta keys to be saved in the
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
                 task_handlers=dict(),
                 multi_task_fields=('gt_label', ),
                 meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.multi_task_fields = multi_task_fields
        self.meta_keys = meta_keys
        self.task_handlers = defaultdict(
            partial(PackClsInputs, meta_keys=meta_keys))
        for task_name, task_handler in task_handlers.items():
            self.task_handlers[task_name] = TRANSFORMS.build(
                dict(type=task_handler, meta_keys=meta_keys))

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        result = {'img_path': 'a.png', 'gt_label': {'task1': 1, 'task3': 3},
            'img': array([[[  0,   0,   0])
        """
        packed_results = dict()
        results = results.copy()

        if 'img' in results:
            img = results.pop('img')
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)

        task_results = defaultdict(dict)
        for field in self.multi_task_fields:
            if field in results:
                value = results.pop(field)
                for k, v in value.items():
                    task_results[k].update({field: v})

        data_sample = MultiTaskDataSample()
        for task_name, task_result in task_results.items():
            task_handler = self.task_handlers[task_name]
            task_pack_result = task_handler({**results, **task_result})
            data_sample.set_field(task_pack_result['data_samples'], task_name)

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self):
        repr = self.__class__.__name__
        task_handlers = {
            name: handler.__class__.__name__
            for name, handler in self.task_handlers.items()
        }
        repr += f'(task_handlers={task_handlers}, '
        repr += f'multi_task_fields={self.multi_task_fields}, '
        repr += f'meta_keys={self.meta_keys})'
        return repr


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

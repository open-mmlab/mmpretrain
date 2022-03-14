# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self, transforms, skip_type_keys=None):
        assert isinstance(transforms, Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self.skip_type_keys = skip_type_keys

        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            transform_type = t.__class__.__name__
            if self.skip_type_keys is not None and \
                    transform_type in self.skip_type_keys:
                continue

            data = t(data)
            if data is None:
                return None
        return data

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        It is called by an external hook.
        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self.skip_type_keys = skip_type_keys

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string

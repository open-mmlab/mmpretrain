# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class FlipAug:
    """Test-time augmentation with flipping. An example configuration is as
    followed:

    .. code-block::
        flip=True,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    After FlipAug with above configuration, the results are wrapped
    into lists of the same length as followed:
    .. code-block::
        dict(
            img=[...],
            flip=[False, True, False, True]
            ...
        )
    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal", "vertical" and "diagonal". If
            flip_direction is a list, multiple flip augmentations will be
            applied. It has no effect when flip == False. Default:
            "horizontal".
    """

    def __init__(self, transforms, flip=False, flip_direction='horizontal'):
        self.transforms = Compose(transforms)

        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.
        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        flip_args = [(False, None)]
        if self.flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        for flip, direction in flip_args:
            _results = results.copy()
            _results['flip'] = flip
            _results['flip_direction'] = direction
            data = self.transforms(_results)
            aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'flip={self.flip}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str

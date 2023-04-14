# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, List, Union

from mmcv.transforms import BaseTransform, Compose

from mmpretrain.registry import TRANSFORMS

# Define type of transform or transform config
Transform = Union[dict, Callable[[dict], dict]]


@TRANSFORMS.register_module()
class MultiView(BaseTransform):
    """A transform wrapper for multiple views of an image.

    Args:
        transforms (list[dict | callable], optional): Sequence of transform
            object or config dict to be wrapped.
        mapping (dict): A dict that defines the input key mapping.
            The keys corresponds to the inner key (i.e., kwargs of the
            ``transform`` method), and should be string type. The values
            corresponds to the outer keys (i.e., the keys of the
            data/results), and should have a type of string, list or dict.
            None means not applying input mapping. Default: None.
        allow_nonexist_keys (bool): If False, the outer keys in the mapping
            must exist in the input data, or an exception will be raised.
            Default: False.

    Examples:
        >>> # Example 1: MultiViews 1 pipeline with 2 views
        >>> pipeline = [
        >>>     dict(type='MultiView',
        >>>         num_views=2,
        >>>         transforms=[
        >>>             [
        >>>                dict(type='Resize', scale=224))],
        >>>         ])
        >>> ]
        >>> # Example 2: MultiViews 2 pipelines, the first with 2 views,
        >>> # the second with 6 views
        >>> pipeline = [
        >>>     dict(type='MultiView',
        >>>         num_views=[2, 6],
        >>>         transforms=[
        >>>             [
        >>>                dict(type='Resize', scale=224)],
        >>>             [
        >>>                dict(type='Resize', scale=224),
        >>>                dict(type='RandomSolarize')],
        >>>         ])
        >>> ]
    """

    def __init__(self, transforms: List[List[Transform]],
                 num_views: Union[int, List[int]]) -> None:

        if isinstance(num_views, int):
            num_views = [num_views]
        assert isinstance(num_views, List)
        assert len(num_views) == len(transforms)
        self.num_views = num_views

        self.pipelines = []
        for trans in transforms:
            pipeline = Compose(trans)
            self.pipelines.append(pipeline)

        self.transforms = []
        for i in range(len(num_views)):
            self.transforms.extend([self.pipelines[i]] * num_views[i])

    def transform(self, results: dict) -> dict:
        """Apply transformation to inputs.

        Args:
            results (dict): Result dict from previous pipelines.

        Returns:
            dict: Transformed results.
        """
        multi_views_outputs = dict(img=[])
        for trans in self.transforms:
            inputs = copy.deepcopy(results)
            outputs = trans(inputs)

            multi_views_outputs['img'].append(outputs['img'])
        results.update(multi_views_outputs)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + '('
        for i, p in enumerate(self.pipelines):
            repr_str += f'\nPipeline {i + 1} with {self.num_views[i]} views:\n'
            repr_str += str(p)
        repr_str += ')'
        return repr_str

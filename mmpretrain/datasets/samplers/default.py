# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterator

import torch
from mmengine.dataset import DefaultSampler as DefaultSampler_

from mmpretrain.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DefaultSampler(DefaultSampler_):
    """Wrapper for default sampler to support different sub sample type.

    Args:
        subsample_type (str): The method to subsample data. Only two following
            types are supported:
            - 'default': Original torch behavior. Sample the examples one by
                one for each GPU in terms. For instance, 8 examples on 2 GPUs,
                GPU0: [0,2,4,8], GPU1: [1,3,5,7]
            - 'sequential': Subsample all examples to n chunk sequntially.
                For instance, 8 examples on 2 GPUs,
                GPU0: [0,1,2,3], GPU1: [4,5,6,7]
        **kwargs: Other keyword arguments in :class:`mmengine.DefaultSampler`.
    """

    def __init__(self, subsample_type: str = 'default', **kwargs) -> None:
        super().__init__(**kwargs)
        self.subsample_type = subsample_type

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        if self.subsample_type == 'default':
            indices = indices[self.rank:self.total_size:self.world_size]
        elif self.subsample_type == 'sequential':
            num_samples_per_rank = self.total_size // self.world_size
            indices = indices[self.rank *
                              num_samples_per_rank:(self.rank + 1) *
                              num_samples_per_rank]
        else:
            raise ValueError(
                f'subsample_type {self.subsample_type} not supported.')

        return iter(indices)

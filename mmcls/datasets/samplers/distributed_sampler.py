# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

from mmcls.core.utils import sync_random_seed
from mmcls.datasets import SAMPLERS
from mmcls.utils import auto_select_device


@SAMPLERS.register_module()
class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 round_up=True,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        if self.round_up:
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed, device=auto_select_device())

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == self.num_samples

        return iter(indices)

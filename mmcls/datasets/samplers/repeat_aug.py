import math

import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

from mmcls.core.utils import sync_random_seed
from mmcls.datasets import SAMPLERS


@SAMPLERS.register_module()
class RepeatAugSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for
    distributed, with repeated augmentation. It ensures that different each
    augmented version of a sample will be visible to a different process (GPU).
    Heavily based on torch.utils.data.DistributedSampler.

    This sampler was taken from
    https://github.com/facebookresearch/deit/blob/0c4b8f60/samplers.py
    Used in
    Copyright (c) 2015-present, Facebook, Inc.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 num_repeats=3,
                 selected_round=256,
                 selected_ratio=0,
                 seed=0):
        default_rank, default_world_size = get_dist_info()
        rank = default_rank if rank is None else rank
        num_replicas = (
            default_world_size if num_replicas is None else num_replicas)

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # Determine the number of samples to select per epoch for each rank.
        # num_selected logic defaults to be the same as original RASampler
        # impl, but this one can be tweaked
        # via selected_ratio and selected_round args.
        selected_ratio = selected_ratio or num_replicas  # ratio to reduce
        # selected samples by, num_replicas if 0
        if selected_round:
            self.num_selected_samples = int(
                math.floor(
                    len(self.dataset) // selected_round * selected_round /
                    selected_ratio))
        else:
            self.num_selected_samples = int(
                math.ceil(len(self.dataset) / selected_ratio))

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            if self.num_replicas > 1:  # In distributed environment
                # deterministically shuffle based on epoch
                g = torch.Generator()
                # When :attr:`shuffle=True`, this ensures all replicas
                # use a different random ordering for each epoch.
                # Otherwise, the next iteration of this sampler will
                # yield the same ordering.
                g.manual_seed(self.epoch + self.seed)
                indices = torch.randperm(
                    len(self.dataset), generator=g).tolist()
            else:
                indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
        indices = [x for x in indices for _ in range(self.num_repeats)]
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample per rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # return up to num selected samples
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

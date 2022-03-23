# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import OptimizerHook, get_dist_info
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    grads = [
        param.grad.data for param in params
        if param.requires_grad and param.grad is not None
    ]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


class DistOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()


def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.

    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

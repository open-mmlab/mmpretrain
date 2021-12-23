# Adapted from https://github.com/facebookresearch/pycls/blob/f8cd962737e33ce9e19b3083a33551da95c2d9c0/pycls/core/net.py  # noqa: E501
# Original licence: Copyright (c) 2019 Facebook, Inc under the Apache License 2.0  # noqa: E501

import itertools
import logging

import mmcv
import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import EpochBasedRunner, get_dist_info
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log
from torch.nn import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.parallel import DataParallel, DistributedDataParallel


def is_parallel_module(module):
    """Check if a module is a parallel module.

    The following 3 modules (and their subclasses) are regarded as parallel
    modules: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version).
    Args:
        module (nn.Module): The module to be checked.
    Returns:
        bool: True if the input module is a parallel module.
    """
    parallels = (DataParallel, DistributedDataParallel,
                 MMDistributedDataParallel)
    if isinstance(module, parallels):
        return True
    else:
        return False


def scaled_all_reduce(tensors, num_gpus):
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of
    the process group .
    """
    # There is no need for reduction in the single-proc case
    if num_gpus == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / num_gpus)
    return tensors


@torch.no_grad()
def update_bn_stats(model, loader, num_samples=8192, logger=None):
    """Computes precise BN stats on training data.

    the actual num_items is :
      int(num_samples / batch_size / NUM_GPUS) * batch_size * NUM_GPUS

    Attributes:
        model (nn.module): A pytorch NN model.
        loader (DataLoader): PyTorch dataloader._dataloader
    """
    if is_parallel_module(model):
        parallel_module = model
        model = model.module
    else:
        parallel_module = model

    # get dist info
    rank, NUM_GPUS = get_dist_info()
    # Compute the number of minibatches to use
    num_iter = num_samples // (loader.batch_size * NUM_GPUS)
    num_iter = min(num_iter, len(loader))
    # Retrieve the BN layers
    bn_layers = [m for m in model.modules() if isinstance(m, (_BatchNorm))]

    if len(bn_layers) == 0:
        print_log('No BN found in model', logger=logger, level=logging.WARNING)
        return
    print_log(
        f'{len(bn_layers)} BN found, run {num_iter} iters...', logger=logger)

    # Finds all the other norm layers with training=True.
    for m in model.modules():
        if m.training and isinstance(m, (_InstanceNorm, GroupNorm)):
            print_log(
                'IN/GN stats will not be updated in PreciseHook.',
                logger=logger,
                level=logging.WARNING)

    # Initialize BN stats storage for computing
    # mean(mean(batch)) and mean(var(batch))
    running_means = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_vars = [torch.zeros_like(bn.running_var) for bn in bn_layers]
    # Remember momentum values
    momentums = [bn.momentum for bn in bn_layers]
    # Set momentum to 1.0 to compute BN stats that reflect the current batch
    for bn in bn_layers:
        bn.momentum = 1.0
    # Average the BN stats for each BN layer over the batches
    if rank == 0:
        prog_bar = mmcv.ProgressBar(num_iter)

    for data in itertools.islice(loader, num_iter):
        parallel_module(**data)
        for i, bn in enumerate(bn_layers):
            running_means[i] += bn.running_mean / num_iter
            running_vars[i] += bn.running_var / num_iter
        if rank == 0:
            prog_bar.update()

    # Sync BN stats across GPUs (no reduction if 1 GPU used)
    running_means = scaled_all_reduce(running_means, NUM_GPUS)
    running_vars = scaled_all_reduce(running_vars, NUM_GPUS)
    # Set BN stats and restore original momentum values
    for i, bn in enumerate(bn_layers):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]


@HOOKS.register_module()
class PreciseBNHook(Hook):
    """Precise BN hook.

    Attributes:
        num_items (int): Number of iterations to update the bn stats.
            Default: 8192.
        interval (int): Perform precise bn interval. Default: 1.
    """

    def __init__(self, num_items=8192, interval=1):
        assert interval > 0 and num_items > 0

        self.interval = interval
        self.num_items = num_items

    def _perform_precise_bn(self, runner):
        print_log(
            f'Running Precise BN for {self.num_items} items...',
            logger=runner.logger)
        update_bn_stats(
            runner.model,
            runner.data_loader,
            self.num_items,
            logger=runner.logger)
        print_log(
            'Finish Precise BN, BN stats updated..', logger=runner.logger)

    def after_train_epoch(self, runner):
        """Calculate prcise BN and broadcast BN stats across GPUs.

        Args:
            runner (obj:`EpochBasedRunner`): runner object.
        """
        assert isinstance(runner,
                          EpochBasedRunner), 'Only support `EpochBasedRunner`'

        # if by epoch, do perform precise every `self.interval` epochs;
        if self.every_n_epochs(runner, self.interval):
            self._perform_precise_bn(runner)

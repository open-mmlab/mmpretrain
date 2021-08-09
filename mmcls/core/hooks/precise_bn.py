# Adapted from https://github.com/facebookresearch/pycls/blob/f8cd962737e33ce9e19b3083a33551da95c2d9c0/pycls/core/net.py  # noqa: E501
# Original licence: Copyright (c) 2019 Facebook, Inc under the Apache License 2.0  # noqa: E501

import logging
import time

import mmcv
import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import Hook
from mmcv.utils import print_log
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader


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


def scaled_all_reduce(tensors, NUM_GPUS):
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of
    the process group .
    """
    # There is no need for reduction in the single-proc case
    if NUM_GPUS == 1:
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
        tensor.mul_(1.0 / NUM_GPUS)
    return tensors


@torch.no_grad()
def update_bn_stats(model,
                    loaders,    
                    NUM_GPUS,
                    NUM_SAMPLES_PRECISE=8192,
                    logger=None):
    """Computes precise BN stats on training data.
    
    the actual num_items is :
      int(NUM_SAMPLES_PRECISE / batch_size / NUM_GPUS) * batch_size * NUM_GPUS

    Attributes:
        model (): A pytorch NN model.
        loaders (List[DataLoader]): A List of PyTorch dataloader.
        num_gpus (int): Number of GPUs of whole exp.
        num_items (int): Number of iterations to update the bn stats.
            Default: 8192.
        logger : the logger.
    
    """

    if is_parallel_module(model):
        parallel_module = model
        model = model.module
    else:
        parallel_module = model

    # Compute the number of minibatches to use
    num_iter = int(NUM_SAMPLES_PRECISE / loaders[0].batch_size / NUM_GPUS)
    num_iter = min(num_iter, sum([len(loader) for loader in loaders]))
    # Retrieve the BN layers
    bn_layers = [
        m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)
    ]

    if len(bn_layers) == 0:
        print_log('No BN found in model', logger=logger, level=logging.WARNING)
        return
    print_log(f'{len(bn_layers)} BN found', logger=logger)

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
    prog_bar = mmcv.ProgressBar(num_iter)
    count = 0
    finish = False
    for loader in loaders:
        for data in loader:
            parallel_module(**data)
            for i, bn in enumerate(bn_layers):
                running_means[i] += bn.running_mean / num_iter
                running_vars[i] += bn.running_var / num_iter
            count += 1
            prog_bar.update()
            if count >= num_iter:
                finish = True
                break
        if finish:
            break
    # Sync BN stats across GPUs (no reduction if 1 GPU used)
    running_means = scaled_all_reduce(running_means, NUM_GPUS)
    running_vars = scaled_all_reduce(running_vars, NUM_GPUS)
    # Set BN stats and restore original momentum values
    for i, bn in enumerate(bn_layers):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]


class PreciseBNHook(Hook):
    """Precise BN hook.

    Attributes:
        dataloaders (DataLoader): A List of PyTorch dataloader.
        num_gpus (int): Number of GPUs of whole exp.
        num_items (int): Number of iterations to update the bn stats.
            Default: 8192.
        interval (int): Perform precise bn interval (by epochs). Default: 1.
    """

    def __init__(self, dataloaders, num_gpus, num_items=8192, interval=1):
        assert len(dataloaders) >= 0, "dataloaders is empty..."
        if not isinstance(dataloaders, list):
            raise TypeError('dataloaders must be a List ,but got', 
                            f' {type(dataloaders)}')
        if not isinstance(dataloaders[0], DataLoader):
            raise TypeError('dataloaders must be a Pytorch Dataloader ,but got', 
                            f' {type(dataloaders)}')
        self.dataloaders = dataloaders
        self.interval = interval
        self.num_items = num_items
        self.BUM_GPUS = num_gpus

    def after_train_epoch(self, runner):
        if self.every_n_epochs(runner, self.interval):
            # sleep to avoid possible deadlock
            time.sleep(2.)
            print_log(
                f'Running Precise BN for {self.num_items} items...',
                logger=runner.logger)
            update_bn_stats(
                runner.model,
                self.dataloaders,
                self.num_items,
                self.BUM_GPUS,
                logger=runner.logger)
            print_log('Finish Precise BN, BN stats updated..', 
                    logger=runner.logger)
            # sleep to avoid possible deadlock
            time.sleep(2.)

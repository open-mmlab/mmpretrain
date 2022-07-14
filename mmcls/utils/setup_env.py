# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmengine import DefaultScope


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Defaults to None.
        device (str): The device where the seed will be put on.
            Defaults to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmcls into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmcls default scope.
            If True, the global default scope will be set to `mmcls`, and all
            registries will build modules from mmcls's registry node. To
            understand more about the registry, please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmcls.data  # noqa: F401,F403
    import mmcls.datasets  # noqa: F401,F403
    import mmcls.engine  # noqa: F401,F403
    import mmcls.metrics  # noqa: F401,F403
    import mmcls.models  # noqa: F401,F403
    import mmcls.visualization  # noqa: F401,F403

    if not init_default_scope:
        return

    current_scope = DefaultScope.get_current_instance()
    if current_scope is None:
        DefaultScope.get_instance('mmcls', scope_name='mmcls')
    elif current_scope.scope_name != 'mmcls':
        warnings.warn(f'The current default scope "{current_scope.scope_name}"'
                      ' is not "mmcls", `register_all_modules` will force the '
                      'current default scope to be "mmcls". If this is not '
                      'expected, please set `init_default_scope=False`.')
        # avoid name conflict
        new_instance_name = f'mmcls-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='mmcls')

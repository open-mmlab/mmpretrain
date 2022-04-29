# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, Fp16OptimizerHook,
                         build_optimizer, build_runner, get_dist_info)
from mmcv.runner.hooks import DistEvalHook, EvalHook

from mmcls.core import DistOptimizerHook
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import get_root_logger


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

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


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                device=None,
                meta=None):
    """Train a model.

    This method will build dataloaders, wrap the model and build a runner
    according to the provided config.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        dataset (:obj:`mmcls.datasets.BaseDataset` | List[BaseDataset]):
            The dataset used to train the model. It can be a single dataset,
            or a list of dataset with the same length as workflow.
        cfg (:obj:`mmcv.utils.Config`): The configs of the experiment.
        distributed (bool): Whether to train the model in a distributed
            environment. Defaults to False.
        validate (bool): Whether to do validation with
            :obj:`mmcv.runner.EvalHook`. Defaults to False.
        timestamp (str, optional): The timestamp string to auto generate the
            name of log files. Defaults to None.
        device (str, optional): TODO
        meta (dict, optional): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
    """
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=cfg.ipu_replicas if device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        if device == 'cpu':
            warnings.warn(
                'The argument `device` is deprecated. To use cpu to train, '
                'please refers to https://mmclassification.readthedocs.io/en'
                '/latest/getting_started.html#train-a-model')
            model = model.cpu()
        elif device == 'ipu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            if not model.device_ids:
                from mmcv import __version__, digit_version
                assert digit_version(__version__) >= (1, 4, 4), \
                    'To train with CPU, please confirm your mmcv version ' \
                    'is not lower than v1.4.4'

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    if device == 'ipu':
        if not cfg.runner['type'].startswith('IPU'):
            cfg.runner['type'] = 'IPU' + cfg.runner['type']
        if 'options_cfg' not in cfg.runner:
            cfg.runner['options_cfg'] = {}
        cfg.runner['options_cfg']['replicationFactor'] = cfg.ipu_replicas
        cfg.runner['fp16_cfg'] = cfg.get('fp16', None)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        if device == 'ipu':
            from mmcv.device.ipu import IPUFp16OptimizerHook
            optimizer_config = IPUFp16OptimizerHook(
                **cfg.optimizer_config,
                loss_scale=fp16_cfg['loss_scale'],
                distributed=distributed)
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config,
                loss_scale=fp16_cfg['loss_scale'],
                distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    if distributed and cfg.runner['type'] == 'EpochBasedRunner':
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            'shuffle': False,  # Not shuffle by default
            'sampler_cfg': None,  # Not use sampler by default
            **cfg.data.get('val_dataloader', {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # `EvalHook` needs to be executed after `IterTimerHook`.
        # Otherwise, it will cause a bug if use `IterBasedRunner`.
        # Refers to https://github.com/open-mmlab/mmcv/issues/1261
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

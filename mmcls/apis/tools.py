import copy
import mmcv
import numpy as np
import os.path as osp
import time
import torch
import warnings
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from .. import __version__
from ..core import wrap_fp16_model
from ..datasets import build_dataloader, build_dataset
from ..models import build_classifier
from ..utils import collect_env, get_root_logger
from . import multi_gpu_test, set_random_seed, single_gpu_test, train_model


def test(config,
         checkpoint,
         out=None,
         metric='accuracy',
         gpu_collect=True,
         options=None,
         launcher='none',
         local_rank=0):
    """High-level API for testing model.

    This method is a function equivalence to command line tool/test.py.

    Args:
        config (str): Path to test config file.
        checkpoint (str): Path to checkpoint file.
        out (str, optional): Path to dump the result.
        metric (str, optional): Evaluation metric.
        gpu_collect (bool, optional): Whether to use gpu to collect results.
        options (dict, optional): Additional options, will merge to or
            override some settings provided from config file.
        launcher (str, optional): job launcher.
        local_rank (int): Local rank.

    Returns:
        dict: The prediction results. Available only for rank 0.
    """
    # Store all arguments into args
    args = type('', (), {})()
    args.config = config
    args.checkpoint = checkpoint
    args.out = out
    args.metric = metric
    args.gpu_collect = gpu_collect
    args.options = options
    args.launcher = launcher
    args.local_rank = local_rank

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.metric != '':
            results = dataset.evaluate(outputs, args.metric)
        else:
            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            if 'CLASSES' in checkpoint['meta']:
                CLASSES = checkpoint['meta']['CLASSES']
            else:
                from mmcls.datasets import ImageNet
                warnings.simplefilter('once')
                warnings.warn('Class names are not saved in the checkpoint\'s '
                              'meta data, use imagenet by default.')
                CLASSES = ImageNet.CLASSES
            pred_class = [CLASSES[lb] for lb in pred_label]
            results = {
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_class': pred_class
            }
            # remove prints
            if args.out and rank == 0:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(results, args.out)

        return results


def train(config,
          work_dir=None,
          resume_from=None,
          no_validate=False,
          gpus=1,
          gpu_ids=range(1),
          seed=None,
          deterministic=False,
          options=None,
          launcher='none',
          local_rank=0,
          autoscale_lr=False):
    """High-level API for training model.

    This method is a function equivalence to command line tool/train.py.

    Args:
        config (str): path to train config file.
        work_dir (str, optional): directory path to save logs and models.
        resume_from (str, optional): path to checkpoint file to resume from.
        no_validate (bool, optional): whether not to evaluate the checkpoint during training
        gpus (int, optional): number of gpus to use (only applicable to non-distributed training)
        gpu_ids (int, optional): ids of gpus to use (only applicable to non-distributed training)
        seed (int, optional): random seed.
        deterministic (bool, optional): whether to set deterministic options for CUDNN backend.
        options (dict, optional): Additional options, will merge to or override some settings provided from config file.
        launcher (str, optional): job launcher.
        local_rank (int): Local rank.
        autoscale_lr (bool, optional): Automatically scale lr with the number of gpus.
    """
    # Store all arguments into args
    args = type('', (), {})()
    args.config = config
    args.work_dir = work_dir
    args.resume_from = resume_from
    args.no_validate = no_validate
    args.gpus = gpus
    args.gpu_ids = gpu_ids
    args.seed = seed
    args.deterministic = deterministic
    args.gpu_ids = gpu_ids
    args.options = options
    args.launcher = launcher
    args.local_rank = local_rank
    args.autoscale_lr = autoscale_lr

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_classifier(cfg.model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmcls version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
from datetime import datetime
from pathlib import Path

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmcls import __version__
from mmcls.apis import init_random_seed, set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger, load_json_log

TEST_METRICS = ('precision', 'recall', 'f1_score', 'support', 'mAP', 'CP',
                'CR', 'CF1', 'OP', 'OR', 'OF1', 'accuracy')

prog_description = """K-Fold cross-validation.

To start a 5-fold cross-validation experiment:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5

To resume a 5-fold cross-validation from an interrupted experiment:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5 --resume-from work_dirs/fold2/latest.pth

To summarize a 5-fold cross-validation:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5 --summary
"""  # noqa: E501


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=prog_description)
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-splits', type=int, help='The number of all folds.')
    parser.add_argument(
        '--fold',
        type=int,
        help='The fold used to do validation. '
        'If specify, only do an experiment of the specified fold.')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Summarize the k-fold cross-validation results.')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def copy_config(old_cfg):
    """deepcopy a Config object."""
    new_cfg = Config()
    _cfg_dict = copy.deepcopy(old_cfg._cfg_dict)
    _filename = copy.deepcopy(old_cfg._filename)
    _text = copy.deepcopy(old_cfg._text)
    super(Config, new_cfg).__setattr__('_cfg_dict', _cfg_dict)
    super(Config, new_cfg).__setattr__('_filename', _filename)
    super(Config, new_cfg).__setattr__('_text', _text)
    return new_cfg


def train_single_fold(args, cfg, fold, distributed, seed):
    # create the work_dir for the fold
    work_dir = osp.join(cfg.work_dir, f'fold{fold}')
    cfg.work_dir = work_dir

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # wrap the dataset cfg
    train_dataset = dict(
        type='KFoldDataset',
        fold=fold,
        dataset=cfg.data.train,
        num_splits=args.num_splits,
        seed=seed,
    )
    val_dataset = dict(
        type='KFoldDataset',
        fold=fold,
        # Use the same dataset with training.
        dataset=copy.deepcopy(cfg.data.train),
        num_splits=args.num_splits,
        seed=seed,
        test_mode=True,
    )
    val_dataset['dataset']['pipeline'] = cfg.data.val.pipeline
    cfg.data.train = train_dataset
    cfg.data.val = val_dataset
    cfg.data.test = val_dataset

    # dump config
    stem, suffix = osp.basename(args.config).rsplit('.', 1)
    cfg.dump(osp.join(cfg.work_dir, f'{stem}_fold{fold}.{suffix}'))
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
    logger.info(
        f'-------- Cross-validation: [{fold+1}/{args.num_splits}] -------- ')

    # set random seeds
    # Use different seed in different folds
    logger.info(f'Set random seed to {seed + fold}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed + fold, deterministic=args.deterministic)
    cfg.seed = seed + fold
    meta['seed'] = seed + fold

    model = build_classifier(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    meta.update(
        dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            kfold=dict(fold=fold, num_splits=args.num_splits)))
    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        device='cpu' if args.device == 'cpu' else 'cuda',
        meta=meta)


def summary(args, cfg):
    summary = dict()
    for fold in range(args.num_splits):
        work_dir = Path(cfg.work_dir) / f'fold{fold}'

        # Find the latest training log
        log_files = list(work_dir.glob('*.log.json'))
        if len(log_files) == 0:
            continue
        log_file = sorted(log_files)[-1]

        date = datetime.fromtimestamp(log_file.lstat().st_mtime)
        summary[fold] = {'date': date.strftime('%Y-%m-%d %H:%M:%S')}

        # Find the latest eval log
        json_log = load_json_log(log_file)
        epochs = sorted(list(json_log.keys()))
        eval_log = {}

        def is_metric_key(key):
            for metric in TEST_METRICS:
                if metric in key:
                    return True
            return False

        for epoch in epochs[::-1]:
            if any(is_metric_key(k) for k in json_log[epoch].keys()):
                eval_log = json_log[epoch]
                break

        summary[fold]['epoch'] = epoch
        summary[fold]['metric'] = {
            k: v[0]  # the value is a list with only one item.
            for k, v in eval_log.items() if is_metric_key(k)
        }
    show_summary(args, summary)


def show_summary(args, summary_data):
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        raise ImportError('Please run `pip install rich` to install '
                          'package `rich` to draw the table.')

    console = Console()
    table = Table(title=f'{args.num_splits}-fold Cross-validation Summary')
    table.add_column('Fold')
    metrics = summary_data[0]['metric'].keys()
    for metric in metrics:
        table.add_column(metric)
    table.add_column('Epoch')
    table.add_column('Date')

    for fold in range(args.num_splits):
        row = [f'{fold+1}']
        if fold not in summary_data:
            table.add_row(*row)
            continue
        for metric in metrics:
            metric_value = summary_data[fold]['metric'].get(metric, '')

            def format_value(value):
                if isinstance(value, float):
                    return f'{value:.2f}'
                if isinstance(value, (list, tuple)):
                    return str([format_value(i) for i in value])
                else:
                    return str(value)

            row.append(format_value(metric_value))
        row.append(str(summary_data[fold]['epoch']))
        row.append(summary_data[fold]['date'])
        table.add_row(*row)

    console.print(table)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
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

    if args.summary:
        summary(args, cfg)
        return

    # resume from the previous experiment
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        resume_kfold = torch.load(cfg.resume_from).get('meta',
                                                       {}).get('kfold', None)
        if resume_kfold is None:
            raise RuntimeError(
                'No "meta" key in checkpoints or no "kfold" in the meta dict. '
                'Please check if the resume checkpoint from a k-fold '
                'cross-valid experiment.')
        resume_fold = resume_kfold['fold']
        assert args.num_splits == resume_kfold['num_splits']
    else:
        resume_fold = 0

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # init a unified random seed
    seed = init_random_seed(args.seed)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = range(resume_fold, args.num_splits)

    for fold in folds:
        cfg_ = copy_config(cfg)
        if fold != resume_fold:
            cfg_.resume_from = None
        train_single_fold(args, cfg_, fold, distributed, seed)

    if args.fold is None:
        summary(args, cfg)


if __name__ == '__main__':
    main()

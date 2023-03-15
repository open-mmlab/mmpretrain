# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.dist import sync_random_seed
from mmengine.fileio import dump, load
from mmengine.hooks import Hook
from mmengine.runner import Runner, find_latest_checkpoint
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

EXP_INFO_FILE = 'kfold_exp.json'

prog_description = """K-Fold cross-validation.

To start a 5-fold cross-validation experiment:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5

To resume a 5-fold cross-validation from an interrupted experiment:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5 --resume
"""  # noqa: E501


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=prog_description)
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-splits',
        type=int,
        help='The number of all folds.',
        required=True)
    parser.add_argument(
        '--fold',
        type=int,
        help='The fold used to do validation. '
        'If specify, only do an experiment of the specified fold.')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume the previous experiment.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
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


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set dataloader args
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
        default_dataloader_cfg.persistent_workers = False

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = copy.deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False
        if args.no_persistent_workers:
            cfg[field]['persistent_workers'] = False

    set_default_dataloader_cfg(cfg, 'train_dataloader')
    set_default_dataloader_cfg(cfg, 'val_dataloader')
    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def train_single_fold(cfg, num_splits, fold, resume_ckpt=None):
    root_dir = cfg.work_dir
    cfg.work_dir = osp.join(root_dir, f'fold{fold}')
    if resume_ckpt is not None:
        cfg.resume = True
        cfg.load_from = resume_ckpt
    dataset = cfg.train_dataloader.dataset

    # wrap the dataset cfg
    def wrap_dataset(dataset, test_mode):
        return dict(
            type='KFoldDataset',
            dataset=dataset,
            fold=fold,
            num_splits=num_splits,
            seed=cfg.kfold_split_seed,
            test_mode=test_mode,
        )

    train_dataset = copy.deepcopy(dataset)
    cfg.train_dataloader.dataset = wrap_dataset(train_dataset, False)

    if cfg.val_dataloader is not None:
        if 'pipeline' not in cfg.val_dataloader.dataset:
            raise ValueError(
                'Cannot find `pipeline` in the validation dataset. '
                "If you are using dataset wrapper, please don't use this "
                'tool to act kfold cross validation. '
                'Please write config files manually.')
        val_dataset = copy.deepcopy(dataset)
        val_dataset['pipeline'] = cfg.val_dataloader.dataset.pipeline
        cfg.val_dataloader.dataset = wrap_dataset(val_dataset, True)
    if cfg.test_dataloader is not None:
        if 'pipeline' not in cfg.test_dataloader.dataset:
            raise ValueError(
                'Cannot find `pipeline` in the test dataset. '
                "If you are using dataset wrapper, please don't use this "
                'tool to act kfold cross validation. '
                'Please write config files manually.')
        test_dataset = copy.deepcopy(dataset)
        test_dataset['pipeline'] = cfg.test_dataloader.dataset.pipeline
        cfg.test_dataloader.dataset = wrap_dataset(test_dataset, True)

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.logger.info(
        f'----------- Cross-validation: [{fold+1}/{num_splits}] ----------- ')
    runner.logger.info(f'Train dataset: \n{runner.train_dataloader.dataset}')

    class SaveInfoHook(Hook):

        def after_train_epoch(self, runner):
            last_ckpt = find_latest_checkpoint(cfg.work_dir)
            exp_info = dict(
                fold=fold,
                last_ckpt=last_ckpt,
                kfold_split_seed=cfg.kfold_split_seed,
            )
            dump(exp_info, osp.join(root_dir, EXP_INFO_FILE))

    runner.register_hook(SaveInfoHook(), 'LOWEST')

    # start training
    runner.train()


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # set the unify random seed
    cfg.kfold_split_seed = args.seed or sync_random_seed()

    # resume from the previous experiment
    if args.resume:
        experiment_info = load(osp.join(cfg.work_dir, EXP_INFO_FILE))
        resume_fold = experiment_info['fold']
        cfg.kfold_split_seed = experiment_info['kfold_split_seed']
        resume_ckpt = experiment_info.get('last_ckpt', None)
    else:
        resume_fold = 0
        resume_ckpt = None

    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = range(resume_fold, args.num_splits)

    for fold in folds:
        cfg_ = copy.deepcopy(cfg)
        train_single_fold(cfg_, args.num_splits, fold, resume_ckpt)
        resume_ckpt = None


if __name__ == '__main__':
    main()

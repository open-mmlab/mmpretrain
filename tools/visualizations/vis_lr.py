# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import re
import time
from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import mmcv
import torch.nn as nn
from mmcv import Config, DictAction, ProgressBar
from mmcv.runner import (EpochBasedRunner, IterBasedRunner, IterLoader,
                         build_optimizer)
from torch.utils.data import DataLoader

from mmcls.utils import get_root_logger


class DummyEpochBasedRunner(EpochBasedRunner):
    """Fake Epoch-based Runner.

    This runner won't train model, and it will only call hooks and return all
    learning rate in each iteration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar = ProgressBar(self._max_epochs, start=False)

    def train(self, data_loader, **kwargs):
        lr_list = []
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        for i in range(len(self.data_loader)):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            lr_list.append(self.current_lr())
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
        self.progress_bar.update(1)
        return lr_list

    def run(self, data_loaders, workflow, **kwargs):
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        self.progress_bar.start()
        lr_list = []
        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    lr_list.extend(epoch_runner(data_loaders[i], **kwargs))

        self.progress_bar.file.write('\n')
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        return lr_list


class DummyIterBasedRunner(IterBasedRunner):
    """Fake Iter-based Runner.

    This runner won't train model, and it will only call hooks and return all
    learning rate in each iteration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar = ProgressBar(self._max_iters, start=False)

    def train(self, data_loader, **kwargs):
        lr_list = []
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        next(data_loader)
        self.call_hook('before_train_iter')
        lr_list.append(self.current_lr())
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1
        self.progress_bar.update(1)
        return lr_list

    def run(self, data_loaders, workflow, **kwargs):
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        self.progress_bar.start()
        lr_list = []
        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    lr_list.extend(iter_runner(iter_loaders[i], **kwargs))

        self.progress_bar.file.write('\n')
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')
        return lr_list


class SimpleModel(nn.Module):
    """simple model that do nothing in train_step."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def train_step(self, *args, **kwargs):
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize a Dataset Pipeline')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--dataset-size',
        type=int,
        help='The size of the dataset. If specify, `build_dataset` will '
        'be skipped and use this size as the dataset size.')
    parser.add_argument(
        '--ngpus',
        type=int,
        default=1,
        help='The number of GPUs used in training.')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--style', type=str, default='whitegrid', help='style of plt')
    parser.add_argument(
        '--save-path',
        type=Path,
        help='The learning rate curve plot save path')
    parser.add_argument(
        '--window-size',
        default='12*7',
        help='Size of the window to display images, in format of "$W*$H".')
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
    args = parser.parse_args()
    if args.window_size != '':
        assert re.match(r'\d+\*\d+', args.window_size), \
            "'window-size' must be in format 'W*H'."

    return args


def plot_curve(lr_list, args, iters_per_epoch, by_epoch=True):
    """Plot learning rate vs iter graph."""
    try:
        import seaborn as sns
        sns.set_style(args.style)
    except ImportError:
        print("Attention: The plot style won't be applied because 'seaborn' "
              'package is not installed, please install it if you want better '
              'show style.')
    wind_w, wind_h = args.window_size.split('*')
    wind_w, wind_h = int(wind_w), int(wind_h)
    plt.figure(figsize=(wind_w, wind_h))
    # if legend is None, use {filename}_{key} as legend

    ax: plt.Axes = plt.subplot()

    ax.plot(lr_list, linewidth=1)
    if by_epoch:
        ax.xaxis.tick_top()
        ax.set_xlabel('Iters')
        ax.xaxis.set_label_position('top')
        sec_ax = ax.secondary_xaxis(
            'bottom',
            functions=(lambda x: x / iters_per_epoch,
                       lambda y: y * iters_per_epoch))
        sec_ax.set_xlabel('Epochs')
        #  ticks = range(0, len(lr_list), iters_per_epoch)
        #  plt.xticks(ticks=ticks, labels=range(len(ticks)))
    else:
        plt.xlabel('Iters')
    plt.ylabel('Learning Rate')

    if args.title is None:
        plt.title(f'{osp.basename(args.config)} Learning Rate curve')
    else:
        plt.title(args.title)

    if args.save_path:
        plt.savefig(args.save_path)
        print(f'The learning rate graph is saved at {args.save_path}')
    plt.show()


def simulate_train(data_loader, cfg, by_epoch=True):
    # build logger, data_loader, model and optimizer
    logger = get_root_logger()
    data_loaders = [data_loader]
    model = SimpleModel()
    optimizer = build_optimizer(model, cfg.optimizer)

    # build runner
    if by_epoch:
        runner = DummyEpochBasedRunner(
            max_epochs=cfg.runner.max_epochs,
            model=model,
            optimizer=optimizer,
            logger=logger)
    else:
        runner = DummyIterBasedRunner(
            max_iters=cfg.runner.max_iters,
            model=model,
            optimizer=optimizer,
            logger=logger)

    # register hooks
    runner.register_training_hooks(
        lr_config=cfg.lr_config,
        custom_hooks_config=cfg.get('custom_hooks', None),
    )

    # only use the first train workflow
    workflow = cfg.workflow[:1]
    assert workflow[0][0] == 'train'
    return runner.run(data_loaders, cfg.workflow)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # make sure save_root exists
    if args.save_path and not args.save_path.parent.exists():
        raise Exception(f'The save path is {args.save_path}, and directory '
                        f"'{args.save_path.parent}' do not exist.")

    # init logger
    logger = get_root_logger(log_level=cfg.log_level)
    logger.info('Lr config : \n\n' + pformat(cfg.lr_config, sort_dicts=False) +
                '\n')

    by_epoch = True if cfg.runner.type == 'EpochBasedRunner' else False

    # prepare data loader
    batch_size = cfg.data.samples_per_gpu * args.ngpus

    if args.dataset_size is None and by_epoch:
        from mmcls.datasets.builder import build_dataset
        dataset_size = len(build_dataset(cfg.data.train))
    else:
        dataset_size = args.dataset_size or batch_size

    fake_dataset = list(range(dataset_size))
    data_loader = DataLoader(fake_dataset, batch_size=batch_size)
    dataset_info = (f'\nDataset infos:'
                    f'\n - Dataset size: {dataset_size}'
                    f'\n - Samples per GPU: {cfg.data.samples_per_gpu}'
                    f'\n - Number of GPUs: {args.ngpus}'
                    f'\n - Total batch size: {batch_size}')
    if by_epoch:
        dataset_info += f'\n - Iterations per epoch: {len(data_loader)}'
    logger.info(dataset_info)

    # simulation training process
    lr_list = simulate_train(data_loader, cfg, by_epoch)

    plot_curve(lr_list, args, len(data_loader), by_epoch)


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import fcntl
import os
import time
from multiprocessing import Pool
from pathlib import Path

from mmcv import Config, DictAction

from mmcls.datasets import build_dataset
from mmcls.datasets.pipelines import Compose


def parse_args():
    parser = argparse.ArgumentParser(description='Verify Dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--log_file',
        type=str,
        default='./broken_files.log',
        help='log the broken files files')
    parser.add_argument(
        '--num-process', type=int, default=1, help='number of process to use')
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
    assert args.log_file is not None
    assert args.num_process > 0
    return args


class Dataset_vaild():
    """the dataset tool class to check if all file are broken."""

    def __init__(self, dataset_cfg, log_file_path, single_process=True):
        super(Dataset_vaild, self).__init__()
        # keep only LoadImageFromFile pipeline
        dataset_cfg.data.train.pipeline = []
        LoadImageFromFile_pipeline = [dict(type='LoadImageFromFile')]
        pipeline = Compose(LoadImageFromFile_pipeline)
        dataset = build_dataset(dataset_cfg.data.train)

        self.loadFile_pipeline = pipeline
        self.dataset = dataset
        self.log_file_path = log_file_path
        self.task_num = len(self)
        # the number to call print function, if the files is less than
        # 100000, the value is 5, else 100
        self.intervals = 100 if self.task_num > 100000 else 5
        # check number files everytime to print some someting
        self.print_intervals = self.task_num // self.intervals
        self.single_process = single_process
        self.start = time.time()

    def vaild_idx(self, idx):
        if idx % self.print_intervals == 0:
            time_spend = round(time.time() - self.start, 2)
            precent = int(idx / self.print_intervals * 100 / self.intervals)
            files_pre_second = idx // time_spend
            print(f'[{precent}%]\t| {idx} pictures, consumes {time_spend} '
                  f'seconds, around {files_pre_second} files pre second.')
        try:
            item = self.dataset[idx]
            item = self.loadFile_pipeline(item)
        except Exception:
            with open(self.log_file_path, 'a') as f:
                # add file lock to prevent multi-process writing errors
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                filepath = item['img_prefix'] + item['img_info']['filename']
                f.write(filepath + '\n')
                print(f'{filepath} cannot be read correctly, please check it.')
                # Release files lock automatic using with

    def __len__(self):
        return len(self.dataset)


def print_info(log_file_path):
    """print some information and do extra action."""
    print()
    with open(log_file_path, 'r') as f:
        context = f.read().strip()
        if context == '':
            print('There is no broken file found.')
            os.remove(log_file_path)
        else:
            num_file = len(context.split('\n'))
            print(f'{num_file} broken files found, name list save in file:'
                  f'{log_file_path}')
    print()


def main():
    # parse cfg and args
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # touch log file to save broken files list.
    log_file_path = Path(args.log_file)
    if not log_file_path.parent.exists():
        raise Exception('log_file dir not found.')
    if log_file_path.exists():
        os.remove(log_file_path)
    log_file_path.touch()

    # do vaild
    single_process = args.num_process == 1
    dataset_vaild = Dataset_vaild(
        cfg, log_file_path, single_process=single_process)
    with Pool(args.num_process) as p:
        p.map(dataset_vaild.vaild_idx, list(range(len(dataset_vaild))))

    print_info(log_file_path)


if __name__ == '__main__':
    main()

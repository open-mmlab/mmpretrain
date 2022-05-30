# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import itertools
import os.path as osp
import sys

import mmcv
from mmcv import Config, DictAction

from mmcls.datasets.builder import build_dataset
from mmcls.registry import VISUALIZERS
from mmcls.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default='./outputs',
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Default train.')
    parser.add_argument(
        '--show-number',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, must bigger than 0. if '
        'the number is bigger than length of dataset, show all the images in '
        'dataset; default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--rescale-factor',
        type=float,
        help='image rescale factor, which is useful if the output is too '
        'large or too small.')
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
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    register_all_modules()
    dataloader = cfg[f'{args.phase}_dataloader']
    dataset = build_dataset(dataloader.dataset)

    cfg.visualizer.save_dir = args.output_dir
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    display_number = min(args.show_number, len(dataset))
    progress_bar = mmcv.ProgressBar(display_number)

    for item in itertools.islice(dataset, display_number):
        img = item['inputs'].permute(1, 2, 0).numpy()
        data_sample = item['data_sample'].numpy()
        img_path = osp.basename(item['data_sample'].img_path)

        out_file = osp.join(
            args.output_dir,
            osp.basename(img_path)) if args.output_dir is not None else None

        img = img[..., [2, 1, 0]]  # bgr to rgb
        visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample,
            rescale_factor=args.rescale_factor,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file)

        progress_bar.update()


if __name__ == '__main__':
    main()

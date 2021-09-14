import argparse
import itertools
import os
import re
import sys
from pathlib import Path

import mmcv
from mmcls.core import visualization as vis
import numpy as np
from mmcv import Config, DictAction
from mmcls.datasets.builder import build_dataset
from mmcls.datasets.pipelines import Compose


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize a Dataset Pipeline')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='*',
        default=['ToTensor', 'Normalize', 'ImageToTensor', 'Collect'],
        help='the pipelines to skip when visualization')
    parser.add_argument(
        '--output-dir', 
        default='', 
        type=str, 
        help='folder to save output pictures')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, only accept "train", '
        '"test" or "val".')
    parser.add_argument(
        '--number',
        type=int,
        default=sys.maxint,
        help='number of images selected to visualize, must bigger than 0. if '
        'the number is bigger than length of dataset, show all the images in '
        'dataset; defalut maxint, means to show all images in  dateset')
    parser.add_argument(
        '--mode',
        default='pipeline',
        type=str,
        choices=['original', 'pipeline', 'concat'],
        help='display mode, to display original pictures or transformed'
        ' picture or comparison pictures. "original" means show images from disk'
        '; "pipeline" means to show images after pipeline; "concat" means show '
        'images Stitched by "original" and "pipeline" images.')
    parser.add_argument(
        '--show',
        default=False,
        action='store_true',
        help='whether to display images in pop-up windows')
    parser.add_argument(
        '--bgr2rgb',
        default=False,
        action='store_true',
        help='flip the color channel order of images, eg. from "RBG" to "BGR".')
    parser.add_argument(
        '--original-display-shape',
        default='',
        help='the resolution of original pictures to display, eg. "224*224".')
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

    assert args.number > 0, f"'args.number' must bigger than zero."
    if args.original_display_shape != '':
        assert re.match(r'\d+\*\d+', args.original_display_shape), \
            "'original_display_shape' should like 'W*H'"

    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options, phase):
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    data_cfg = cfg.data[phase]
    while 'dataset' in data_cfg:
        data_cfg = data_cfg['dataset']
    data_cfg['pipeline'] = [
        x for x in data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg

def build_dataset_pipeline(cfg, phase):
    ''' build dataset and pipeline from config.'''
    data_cfg = cfg.data[phase]
    loadimage_pipeline = []
    if len(data_cfg.pipeline
           ) != 0 and data_cfg.pipeline[0]['type'] == 'LoadImageFromFile':
        loadimage_pipeline.append(data_cfg.pipeline.pop(0))
    origin_pipeline = data_cfg.pipeline
    data_cfg.pipeline = loadimage_pipeline
    dataset = build_dataset(data_cfg)
    pipeline = Compose(origin_pipeline)

    return dataset, pipeline
    

def put_img(board, img, center):
    """put a image into a big board image with the anchor center."""
    center_x, center_y = center
    img_h, img_w, _ = img.shape
    board_h, board_w, _ = board.shape
    xmin, ymin = int(center_x - img_w // 2), int(center_y - img_h // 2)
    assert xmin >= 0 and ymin >= 0, 'Cannot exceed the border'
    assert (ymin + img_h) <= board_h, 'Cannot exceed the border'
    assert (xmin + img_w) <= board_w, 'Cannot exceed the border'
    board[ymin:ymin + img_h, xmin:xmin + img_w, :] = img
    return board


def concat(left_img, right_img):
    """Concat two pictures into a single big picture.
    accept two diffenert shape images.
    """
    left_h, left_w, _ = left_img.shape
    right_h, right_w, _ = right_img.shape
    # create a big board to contain images
    board_h, board_w = max(left_h, right_h), max(left_w, right_w)
    board = np.ones([board_h, 2 * board_w, 3], np.uint8) * 255

    put_img(board, left_img, (int(board_w // 2), int(board_h // 2)))
    put_img(board, right_img, (int(board_w // 2) + board_w, int(board_h // 2)))
    return board


def resize(image, resize_shape):
    """resize image to resize_shape."""
    resize_w, resize_h, *_ = resize_shape.split('*')
    resize_h, resize_w = int(resize_h), int(resize_w)
    image = mmcv.imresize(image, (resize_w, resize_h))
    return image


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options,
                            args.phase)
    dataset, pipeline = build_dataset_pipeline(cfg, args.phase)
    class_names = dataset.CLASSES
    display_number = min(args.number, len(dataset)) 

    with vis.ImshowInfosContextManager() as manager:
        for i, item in enumerate(itertools.islice(dataset, display_number)):
            # some datasets do not have filename, such as minist, cifar
            # save image if args.output_dir is not None
            src_path = item.get('filename', '{}.jpg'.format(i))
            filename = Path(src_path).name
            dist_path = os.path.join(args.output_dir, filename) if args.output_dir else None

            label = class_names[item['gt_label']]
            infos = {"label" : label}

            if args.mode in ['original', 'concat']:
                src_image = item['img'].copy()
                if args.original_display_shape:
                    src_image = resize(src_image, args.original_display_shape)
            if args.mode in ['pipeline', 'concat']:
                item = pipeline(item)
                trans_image = item['img']
                trans_image = np.ascontiguousarray(trans_image, dtype=np.uint8)
                if args.bgr2rgb:
                    trans_image = mmcv.bgr2rgb(trans_image)

            # display original images if args.original is True; display tranformed
            # images if args.transform is True; display concat images if both
            # args.original and args.transform are True
            if args.mode == 'concat':
                image = concat(src_image, trans_image)
            elif args.mode == 'original':
                image = src_image
            elif args.mode == 'pipeline':
                image = trans_image

            if args.show:
                manager.put_img_infos(image, infos, dist_path)


if __name__ == '__main__':
    main()
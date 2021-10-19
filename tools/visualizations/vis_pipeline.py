import argparse
import itertools
import os
import re
import sys
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config, DictAction, ProgressBar

from mmcls.core import visualization as vis
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
        help='the pipelines to skip when visualizing')
    parser.add_argument(
        '--output-dir',
        default='',
        type=str,
        help='folder to save output pictures, if not set, do not save.')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Default train.')
    parser.add_argument(
        '--number',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, must bigger than 0. if '
        'the number is bigger than length of dataset, show all the images in '
        'dataset; default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--mode',
        default='concat',
        type=str,
        choices=['original', 'pipeline', 'concat'],
        help='display mode; display original pictures or transformed pictures'
        ' or comparison pictures. "original" means show images load from disk;'
        ' "pipeline" means to show images after pipeline; "concat" means show '
        'images stitched by "original" and "pipeline" images. Default concat.')
    parser.add_argument(
        '--show',
        default=False,
        action='store_true',
        help='whether to display images in pop-up window. Default False.')
    parser.add_argument(
        '--adaptive',
        default=False,
        action='store_true',
        help='whether to automatically adjust the visualization image size')
    parser.add_argument(
        '--min-edge-length',
        default=200,
        type=int,
        help='the min edge length when visualizing images, used when '
        '"--adaptive" is true. Default 200.')
    parser.add_argument(
        '--max-edge-length',
        default=1000,
        type=int,
        help='the max edge length when visualizing images, used when '
        '"--adaptive" is true. Default 1000.')
    parser.add_argument(
        '--bgr2rgb',
        default=False,
        action='store_true',
        help='flip the color channel order of images')
    parser.add_argument(
        '--window-size',
        default='12*7',
        help='size of the window to display images, in format of "$W*$H".')
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
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for display. key-value pair in xxx=yyy. options '
        'in `mmcls.core.visualization.ImshowInfosContextManager.put_img_infos`'
    )
    args = parser.parse_args()

    assert args.number > 0, "'args.number' must be larger than zero."
    if args.window_size != '':
        assert re.match(r'\d+\*\d+', args.window_size), \
            "'window-size' must be in format 'W*H'."
    if args.output_dir == '' and not args.show:
        raise ValueError("if '--output-dir' and '--show' are not set, "
                         'nothing will happen when the program running.')

    if args.show_options is None:
        args.show_options = {}
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
    """build dataset and pipeline from config.

    Separate the pipeline except 'LoadImageFromFile' step if
    'LoadImageFromFile' in the pipeline.
    """
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
    xmin, ymin = int(center_x - img_w // 2), int(center_y - img_h // 2)
    board[ymin:ymin + img_h, xmin:xmin + img_w, :] = img
    return board


def concat(left_img, right_img):
    """Concat two pictures into a single big picture, accepts two images with
    diffenert shapes."""
    GAP = 10
    left_h, left_w, _ = left_img.shape
    right_h, right_w, _ = right_img.shape
    # create a big board to contain images with shape (board_h, board_w*2+10)
    board_h, board_w = max(left_h, right_h), max(left_w, right_w)
    board = np.ones([board_h, 2 * board_w + GAP, 3], np.uint8) * 255

    put_img(board, left_img, (int(board_w // 2), int(board_h // 2)))
    put_img(board, right_img,
            (int(board_w // 2) + board_w + GAP // 2, int(board_h // 2)))
    return board


def adaptive_size(mode, image, min_edge_length, max_edge_length):
    """rescale image if image is too small to put text like cifra."""
    assert min_edge_length >= 0 and max_edge_length >= 0
    assert max_edge_length >= min_edge_length

    image_h, image_w, *_ = image.shape
    image_w = image_w // 2 if mode == 'concat' else image_w

    if image_h < min_edge_length or image_w < min_edge_length:
        image = mmcv.imrescale(
            image, min(min_edge_length / image_h, min_edge_length / image_h))
    if image_h > max_edge_length or image_w > max_edge_length:
        image = mmcv.imrescale(
            image, max(max_edge_length / image_h, max_edge_length / image_w))
    return image


def get_display_img(item, pipeline, mode, bgr2rgb):
    """get image to display."""
    if bgr2rgb:
        item['img'] = mmcv.bgr2rgb(item['img'])
    src_image = item['img'].copy()
    # get transformed picture
    if mode in ['pipeline', 'concat']:
        item = pipeline(item)
        trans_image = item['img']
        trans_image = np.ascontiguousarray(trans_image, dtype=np.uint8)

    if mode == 'concat':
        image = concat(src_image, trans_image)
    elif mode == 'original':
        image = src_image
    elif mode == 'pipeline':
        image = trans_image
    return image


def main():
    args = parse_args()
    wind_w, wind_h = args.window_size.split('*')
    wind_w, wind_h = int(wind_w), int(wind_h)
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options,
                            args.phase)

    dataset, pipeline = build_dataset_pipeline(cfg, args.phase)
    CLASSES = dataset.CLASSES
    display_number = min(args.number, len(dataset))
    progressBar = ProgressBar(display_number)

    with vis.ImshowInfosContextManager(fig_size=(wind_w, wind_h)) as manager:
        for i, item in enumerate(itertools.islice(dataset, display_number)):
            image = get_display_img(item, pipeline, args.mode, args.bgr2rgb)
            if args.adaptive:
                image = adaptive_size(args.mode, image, args.min_edge_length,
                                      args.max_edge_length)

            # dist_path is None as default, means not save pictures
            dist_path = None
            if args.output_dir:
                # some datasets do not have filename, such as cifar, use id
                src_path = item.get('filename', '{}.jpg'.format(i))
                dist_path = os.path.join(args.output_dir, Path(src_path).name)

            infos = dict(label=CLASSES[item['gt_label']])

            manager.put_img_infos(
                image,
                infos,
                font_size=20,
                out_file=dist_path,
                show=args.show,
                **args.show_options)

            progressBar.update()


if __name__ == '__main__':
    main()

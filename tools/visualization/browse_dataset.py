# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import sys

import cv2
import mmcv
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from mmengine.visualization import Visualizer

from mmpretrain.datasets.builder import build_dataset
from mmpretrain.visualization import ClsVisualizer
from mmpretrain.visualization.cls_visualizer import _get_adaptive_scale


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        '-o',
        default=None,
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--phase',
        '-p',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--show-number',
        '-n',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, must bigger than 0. if '
        'the number is bigger than length of dataset, show all the images in '
        'dataset; default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--show-interval',
        '-i',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--mode',
        '-m',
        default='transformed',
        type=str,
        choices=['original', 'transformed', 'concat', 'pipeline'],
        help='display mode; display original pictures or transformed pictures'
        ' or comparison pictures. "original" means show images load from disk'
        '; "transformed" means to show images after transformed; "concat" '
        'means show images stitched by "original" and "output" images. '
        '"pipeline" means show all the intermediate images. '
        'Defaults to "transformed".')
    parser.add_argument(
        '--rescale-factor',
        '-r',
        type=float,
        help='image rescale factor, which is useful if the output is too '
        'large or too small.')
    parser.add_argument(
        '--channel-order',
        '-c',
        default='BGR',
        choices=['BGR', 'RGB'],
        help='The channel order of the showing images, could be "BGR" '
        'or "RGB", Defaults to "BGR".')
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


def make_grid(imgs, names, rescale_factor=None):
    """Concat list of pictures into a single big picture, align height here."""
    vis = Visualizer()

    ori_shapes = [img.shape[:2] for img in imgs]
    if rescale_factor is not None:
        imgs = [mmcv.imrescale(img, rescale_factor) for img in imgs]

    max_height = int(max(img.shape[0] for img in imgs) * 1.1)
    min_width = min(img.shape[1] for img in imgs)
    horizontal_gap = min_width // 10
    img_scale = _get_adaptive_scale((max_height, min_width))

    texts = []
    text_positions = []
    start_x = 0
    for i, img in enumerate(imgs):
        pad_height = (max_height - img.shape[0]) // 2
        pad_width = horizontal_gap // 2
        # make border
        imgs[i] = cv2.copyMakeBorder(
            img,
            pad_height,
            max_height - img.shape[0] - pad_height + int(img_scale * 30 * 2),
            pad_width,
            pad_width,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255))

        texts.append(f'{names[i]}\n{ori_shapes[i]}')
        text_positions.append(
            [start_x + img.shape[1] // 2 + pad_width, max_height])
        start_x += img.shape[1] + horizontal_gap

    display_img = np.concatenate(imgs, axis=1)
    vis.set_image(display_img)
    img_scale = _get_adaptive_scale(display_img.shape[:2])
    vis.draw_texts(
        texts,
        positions=np.array(text_positions),
        font_sizes=img_scale * 7,
        colors='black',
        horizontal_alignments='center',
        font_families='monospace')
    return vis.get_image()


class InspectCompose(Compose):
    """Compose multiple transforms sequentially.

    And record "img" field of all results in one list.
    """

    def __init__(self, transforms, intermediate_imgs):
        super().__init__(transforms=transforms)
        self.intermediate_imgs = intermediate_imgs

    def __call__(self, data):
        if 'img' in data:
            self.intermediate_imgs.append({
                'name': 'original',
                'img': data['img'].copy()
            })

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
            if 'img' in data:
                self.intermediate_imgs.append({
                    'name': t.__class__.__name__,
                    'img': data['img'].copy()
                })
        return data


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope('mmpretrain')  # Use mmpretrain as default scope.

    dataset_cfg = cfg.get(args.phase + '_dataloader').get('dataset')
    dataset = build_dataset(dataset_cfg)

    intermediate_imgs = []
    dataset.pipeline = InspectCompose(dataset.pipeline.transforms,
                                      intermediate_imgs)

    # init visualizer
    cfg.visualizer.pop('type')
    visualizer = ClsVisualizer(**cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    # init visualization image number
    display_number = min(args.show_number, len(dataset))
    progress_bar = ProgressBar(display_number)

    for i, item in zip(range(display_number), dataset):
        rescale_factor = args.rescale_factor
        if args.mode == 'original':
            image = intermediate_imgs[0]['img']
        elif args.mode == 'transformed':
            image = intermediate_imgs[-1]['img']
        elif args.mode == 'concat':
            ori_image = intermediate_imgs[0]['img']
            trans_image = intermediate_imgs[-1]['img']
            image = make_grid([ori_image, trans_image],
                              ['original', 'transformed'], rescale_factor)
            rescale_factor = None
        else:
            image = make_grid([result['img'] for result in intermediate_imgs],
                              [result['name'] for result in intermediate_imgs],
                              rescale_factor)
            rescale_factor = None

        intermediate_imgs.clear()

        data_sample = item['data_samples'].numpy()

        # get filename from dataset or just use index as filename
        if hasattr(item['data_samples'], 'img_path'):
            filename = osp.basename(item['data_samples'].img_path)
        else:
            # some dataset have not image path
            filename = f'{i}.jpg'

        out_file = osp.join(args.output_dir,
                            filename) if args.output_dir is not None else None

        visualizer.add_datasample(
            filename,
            image if args.channel_order == 'RGB' else image[..., ::-1],
            data_sample,
            rescale_factor=rescale_factor,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file)
        progress_bar.update()


if __name__ == '__main__':
    main()

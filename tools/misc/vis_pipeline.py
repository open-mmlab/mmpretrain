import argparse
import itertools
import os
from pathlib import Path

import cv2
import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmcls.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='visualize dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['ToTensor', 'Normalize', 'ImageToTensor', 'Collect'],
        help='skip some useless pipelines')
    parser.add_argument(
        '--output-dir',
        default='tmp',
        type=str,
        help='Only use when "--show" is False, if there is no display interface, you can save it.')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='which phase of dataset to brower, accept "train" or "test" or "val".')
    parser.add_argument(
        '--number',
        type=int,
        default=-1,
        help='number of images to show;'
        ' if number less than 0, show all the images in dataset')
    parser.add_argument(
        '--show',
        default=False,
        action='store_true',
        help='Whether to display a visual image')
    parser.add_argument(
        '--bgr2rgb',
        default=False,
        action='store_true',
        help='to transform a "bgr" image to a "rgb" image, (some pipeline transform will change the channel order).')
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


def put_text(img, texts, text_color=(0, 0, 255), font_scale=0.6, row_width=20):
    ''' write the label info on the image'''
    x, y = 0, int(row_width * 0.75)
    for text in texts:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, font_scale,
                    text_color, 1)
        y += row_width
    return img

def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options, args.phase)
    dataset = build_dataset(cfg.data[args.phase])
    class_names = dataset.CLASSES

    number = min(args.number,
                 len(dataset)) if args.number >= 0 else len(dataset)
    for i, item in enumerate(itertools.islice(dataset, number)):
        # some dataset do not have filename, such as minist, cifar
        try:
            src_path = item['filename']
            filename = Path(src_path).name
        except:
            filename = "{}.jpg".format(i)
        dist_path = os.path.join(args.output_dir, filename)
        labels = [
            label.strip() for label in class_names[item['gt_label']].split(',')
        ]

        trans_image = item['img']
        trans_image = np.ascontiguousarray(trans_image, dtype=np.uint8)
        if args.bgr2rgb:
            trans_image = mmcv.bgr2rgb(trans_image)


        # Only display the label on the image when pictures are large enough
        h, w, _ = trans_image.shape
        if h >= 160 and w >= 160:
            trans_image = put_text(trans_image, labels)

        if args.show:
            print( labels )
            mmcv.imshow(trans_image)
        else:
            mmcv.imwrite(trans_image, dist_path, auto_mkdir=True)


if __name__ == '__main__':
    main()

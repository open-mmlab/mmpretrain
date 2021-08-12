import argparse
import os
import cv2
from pathlib import Path
import itertools

import numpy as np
import mmcv
from mmcv import Config, DictAction
from mmcls.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['ToTensor', 'Normalize', 'ImageToTensor', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default="tmp",
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument(
        '--number', 
        type=int, 
        default=-1, 
        help='number of images to show;'
             ' if number less than 0, show all the images in dataset')
    parser.add_argument(
        '--original', 
        default=False,
        action='store_true',
        help='Whether to visualize the original image')
    parser.add_argument(
        '--transform', 
        default=False,
        action='store_true',
        help='Whether to visualize the transformed image')
    parser.add_argument(
        '--show', 
        default=False, 
        action='store_true',
        help='Whether to display a visual iamge')
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


def retrieve_data_cfg(config_path, skip_type, cfg_options):
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg:
        train_data_cfg = train_data_cfg['dataset']
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg

def put_text(img, texts, text_color=(0, 0, 255),font_scale=0.6, row_width=20):
    x, y = 0, int(row_width * 0.75)
    for text in texts:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color, 1)
        y += row_width
    return img

def put_img(board, img, center):
    center_x, center_y = center
    h, w, _ = img.shape
    xmin, ymin = int(center_x - w // 2), int(center_y - h // 2)
    board[ymin:ymin+h, xmin:xmin+w,:] = img
    return board

def concat(left_img, right_img):
    left_h, left_w, _ = left_img.shape
    right_h, right_w, _ = right_img.shape
    board_h = int(max(left_h, right_h) * 1.1)
    board_w = int(max(left_w, right_w) * 1.1)
    board = np.ones([board_h, 2*board_w, 3], np.uint8) * 255
    put_img(board, left_img, (int(board_w//2), int(board_h//2)))
    put_img(board, right_img, (int(board_w//2)+board_w, int(board_h//2)))
    return board


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)
    dataset = build_dataset(cfg.data.train)
    class_names = dataset.CLASSES

    number = min(args.number, len(dataset)) if args.number >= 0 else len(dataset)
    for item in itertools.islice(dataset, number):
        src_path = item['filename']
        filename = Path(src_path).name
        dist_path = os.path.join(args.output_dir, filename)
        labels = [label.strip() for label in class_names[item['gt_label']].split(",") ]

        if args.original is True:
            src_image = mmcv.imread(src_path) 
            src_image = put_text(src_image, labels) 
        if args.transform is True:
            trans_image = item['img']
            trans_image = np.ascontiguousarray(trans_image, dtype=np.uint8)
            trans_image = put_text(trans_image , labels)
        
        if args.original and args.transform:
            image = concat(src_image, trans_image)
        elif args.original and not args.transform:
            image = src_image
        elif not args.original and args.transform:
            image = trans_image
        else:
            raise("one of args.original and args.transform must be True...")
    
        if args.show:
            mmcv.imshow(image)
        else:
            mmcv.imwrite(image, dist_path, auto_mkdir=True)


if __name__ == '__main__':
    main()
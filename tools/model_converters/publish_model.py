# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import datetime
import hashlib
import shutil
import warnings
from collections import OrderedDict
from pathlib import Path

import torch

import mmpretrain


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    parser.add_argument(
        '--no-ema',
        action='store_true',
        help='Use keys in `ema_state_dict` (no-ema keys).')
    parser.add_argument(
        '--dataset-type',
        type=str,
        help='The type of the dataset. If the checkpoint is converted '
        'from other repository, this option is used to fill the dataset '
        'meta information to the published checkpoint, like "ImageNet", '
        '"CIFAR10" and others.')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file, args):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove unnecessary fields for smaller file size
    for key in ['optimizer', 'param_schedulers', 'hook_msgs', 'message_hub']:
        checkpoint.pop(key, None)

    # For checkpoint converted from the official weight
    if 'state_dict' not in checkpoint:
        checkpoint = dict(state_dict=checkpoint)

    meta = checkpoint.get('meta', {})
    meta.setdefault('mmpretrain_version', mmpretrain.__version__)

    # handle dataset meta information
    if args.dataset_type is not None:
        from mmpretrain.registry import DATASETS
        dataset_class = DATASETS.get(args.dataset_type)
        dataset_meta = getattr(dataset_class, 'METAINFO', {})
    else:
        dataset_meta = {}

    meta.setdefault('dataset_meta', dataset_meta)

    if len(meta['dataset_meta']) == 0:
        warnings.warn('Missing dataset meta information.')

    checkpoint['meta'] = meta

    ema_state_dict = OrderedDict()
    if 'ema_state_dict' in checkpoint:
        for k, v in checkpoint['ema_state_dict'].items():
            # The ema static dict has some extra fields
            if k.startswith('module.'):
                origin_k = k[len('module.'):]
                assert origin_k in checkpoint['state_dict']
                ema_state_dict[origin_k] = v
        del checkpoint['ema_state_dict']
        print('The input checkpoint has EMA weights, ', end='')
        if args.no_ema:
            # The values stored in `ema_state_dict` is original values.
            print('and drop the EMA weights.')
            assert ema_state_dict.keys() <= checkpoint['state_dict'].keys()
            checkpoint['state_dict'].update(ema_state_dict)
        else:
            print('and use the EMA weights.')

    temp_out_file = Path(out_file).with_name('temp_' + Path(out_file).name)
    torch.save(checkpoint, temp_out_file)

    with open(temp_out_file, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:8]
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file

    current_date = datetime.datetime.now().strftime('%Y%m%d')
    final_file = out_file_name + f'_{current_date}-{sha[:8]}.pth'
    shutil.move(temp_out_file, final_file)

    print(f'Successfully generated the publish-ckpt as {final_file}.')


def main():
    args = parse_args()
    out_dir = Path(args.out_file).parent
    if not out_dir.exists():
        raise ValueError(f'Directory {out_dir} does not exist, '
                         'please generate it manually.')
    process_checkpoint(args.in_file, args.out_file, args)


if __name__ == '__main__':
    main()

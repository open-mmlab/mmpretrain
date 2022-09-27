# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_hornet(ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('head'):
            new_k = k.replace('head.', 'head.fc.')
            new_ckpt[new_k] = new_v
            continue
        elif k.startswith('norm'):
            new_k = k.replace('norm.', 'norm3.')
        elif 'gnconv.pws' in k:
            new_k = k.replace('gnconv.pws', 'gnconv.projs')
        elif 'gamma1' in k:
            new_k = k.replace('gamma1', 'gamma1.weight')
        elif 'gamma2' in k:
            new_k = k.replace('gamma2', 'gamma2.weight')
        else:
            new_k = k

        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained van models to mmcls style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = convert_hornet(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

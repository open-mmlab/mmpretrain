# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_efficientFormer(args, ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('head'):
            new_k = k.replace('head.', 'head.head.')
        elif k.startswith('dist_head'):
            new_k = k.replace('dist_head.', 'head.dist_head.')
        elif k.startswith('patch_embed'):
            items = list(k.split('.'))
            if items[1] in ('0', '3'):
                items[1] = '0' if items[1] == '0' else '1'
                items.insert(2, 'conv')
            elif items[1] in ('1', '4'):
                items[1] = '0' if items[1] == '1' else '1'
                items.insert(2, 'bn')
            new_k = 'backbone.' + '.'.join(items)
        elif k.startswith('network.1.') or k.startswith(
                'network.3.') or k.startswith('network.5.'):
            # mv downsapler to stage
            items = list(k.split('.'))
            items[1] = str(int(items[1]) // 2 + 1)
            items = items[:2] + ['0'] + items[2:]
            new_k = 'backbone.' + '.'.join(items)
        elif k.startswith('norm.'):
            items = list(k.split('.'))
            items[0] = items[0] + '3'
            new_k = 'backbone.' + '.'.join(items)
        else:
            items = list(k.split('.'))
            has_down = 1 if items[1] in '246' else 0
            items[1] = str(int(items[1]) // 2)
            items[2] = str(int(items[2]) + has_down)
            if items[-1] == 'layer_scale_1':
                items[-1] = 'ls1.gamma'
            if items[-1] == 'layer_scale_2':
                items[-1] = 'ls2.gamma'
            new_k = 'backbone.' + '.'.join(items)
        print(f' {k} -> {new_k} : {new_v.shape}')
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained models to '
        'MMClassification style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_efficientFormer(args, state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()

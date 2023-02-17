# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch


def convert_levit(args, ckpt):
    new_ckpt = OrderedDict()
    stage = 0
    block = 0
    change = True
    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('head_dist'):
            new_k = k.replace('head_dist.', 'head.head_dist.')
            new_k = new_k.replace('.l.', '.linear.')
            new_ckpt[new_k] = new_v
            continue
        elif k.startswith('head'):
            new_k = k.replace('head.', 'head.head.')
            new_k = new_k.replace('.l.', '.linear.')
            new_ckpt[new_k] = new_v
            continue
        elif k.startswith('patch_embed'):
            new_k = k.replace('patch_embed.',
                              'patch_embed.patch_embed.').replace(
                                  '.c.', '.conv.')
        elif k.startswith('blocks'):
            strs = k.split('.')
            # new_k = k.replace('.c.', '.').replace('.bn.', '.')
            new_k = k
            if '.m.' in k:
                new_k = new_k.replace('.m.0', '.m.linear1')
                new_k = new_k.replace('.m.2', '.m.linear2')
                new_k = new_k.replace('.m.', '.block.')
                change = True
            elif change:
                stage += 1
                block = int(strs[1])
                change = False
            new_k = new_k.replace(
                'blocks.%s.' % (strs[1]),
                'stages.%d.%d.' % (stage, int(strs[1]) - block))
            new_k = new_k.replace('.c.', '.linear.')
        else:
            new_k = k
        # print(new_k)
        new_k = 'backbone.' + new_k
        new_ckpt[new_k] = new_v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMPretrain style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location='cpu')
    checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_levit(args, state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()

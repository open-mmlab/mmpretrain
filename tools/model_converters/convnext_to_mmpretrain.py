# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_convnext(ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('head'):
            new_k = k.replace('head.', 'head.fc.')
            new_ckpt[new_k] = new_v
            continue
        elif k.startswith('stages'):
            if 'dwconv' in k:
                new_k = k.replace('dwconv', 'depthwise_conv')
            elif 'pwconv' in k:
                new_k = k.replace('pwconv', 'pointwise_conv')
            else:
                new_k = k
        elif k.startswith('norm'):
            new_k = k.replace('norm', 'norm3')
        else:
            new_k = k

        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained convnext '
        'models to mmpretrain style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = convert_convnext(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(dict(state_dict=weight), args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

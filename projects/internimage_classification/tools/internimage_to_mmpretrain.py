# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_internimage(ckpt):
    new_ckpt = OrderedDict()
    for k, v in list(ckpt.items()):
        if 'head.' in k and 'conv_head' not in k:
            if 'weight' in k:
                new_k = 'head.fc.weight'
            else:
                new_k = 'head.fc.bias'
        elif 'patch_embed' in k:
            map_fun = {'conv1': '0', 'norm1': '1', 'conv2': '3', 'norm2': '4'}
            new_k = k
            for old, new in map_fun.items():
                new_k = new_k.replace(old, new)
            new_k = 'backbone.' + new_k

        elif 'levels' in k:
            new_k = k.replace('levels', 'layers')
            if 'mlp' in new_k:
                new_k = new_k.replace('fc1', 'layers.0.0')
                new_k = new_k.replace('fc2', 'layers.1')
            new_k = 'backbone.' + new_k
        elif 'clip_projector.cross_dcn.k_bias' in k:
            continue
        else:
            new_k = 'backbone.' + k
        new_v = v
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained convert_internimage '
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

    weight = convert_internimage(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

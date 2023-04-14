# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_deit3(ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('head'):
            new_k = k.replace('head.', 'head.layers.head.')
            new_ckpt[new_k] = new_v
            continue
        elif k.startswith('patch_embed'):
            if 'proj.' in k:
                new_k = k.replace('proj.', 'projection.')
            else:
                new_k = k
        elif k.startswith('blocks'):
            new_k = k.replace('blocks.', 'layers.')
            if 'norm1' in k:
                new_k = new_k.replace('norm1', 'ln1')
            elif 'norm2' in k:
                new_k = new_k.replace('norm2', 'ln2')
            elif 'mlp.fc1' in k:
                new_k = new_k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = new_k.replace('mlp.fc2', 'ffn.layers.1')
            elif 'gamma_1' in k:
                new_k = new_k.replace('gamma_1', 'attn.gamma1.weight')
            elif 'gamma_2' in k:
                new_k = new_k.replace('gamma_2', 'ffn.gamma2.weight')
        elif k.startswith('norm'):
            new_k = k.replace('norm', 'ln1')
        else:
            new_k = k

        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained deit3 '
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

    weight = convert_deit3(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

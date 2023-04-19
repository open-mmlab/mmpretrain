# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch


def convert_levit(args, ckpt):
    new_ckpt = OrderedDict()
    for k, v in list(ckpt.items()):
        old_k = k
        if k.startswith('backbone'):
            k = k[9:]
        if k.startswith('channel_reduction'):
            continue
        elif k.startswith('patch_embed'):
            k = k.replace('projection', 'proj')
        elif k == "pos_embed":
            B, H, W, C = v.shape
            v = v.reshape(B, H * W, C)
        elif k.startswith('layers'):
            k = k.replace('layers', 'blocks', 1) 
            k = k.replace('ln1', 'norm1')
            k = k.replace('ln2', 'norm2')
            k = k.replace('ffn.layers.0.0', 'mlp.fc1')
            k = k.replace('ffn.layers.1', 'mlp.fc2')
        else:
            k = k
        k = 'backbone.net.' + k
        print(f"{old_k}  -->  {k}  ({v.shape})")
        new_ckpt[k] = v

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

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_davit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('patch_embeds.0'):
            new_k = k.replace('patch_embeds.0', 'patch_embed')
            new_k = new_k.replace('proj', 'projection')
        elif k.startswith('patch_embeds'):
            if k.startswith('patch_embeds.1'):
                new_k = k.replace('patch_embeds.1', 'stages.0.downsample')
            elif k.startswith('patch_embeds.2'):
                new_k = k.replace('patch_embeds.2', 'stages.1.downsample')
            elif k.startswith('patch_embeds.3'):
                new_k = k.replace('patch_embeds.3', 'stages.2.downsample')
            new_k = new_k.replace('proj', 'projection')
        elif k.startswith('main_blocks'):
            new_k = k.replace('main_blocks', 'stages')
            for num_stages in range(4):
                for num_blocks in range(9):
                    if f'{num_stages}.{num_blocks}.0' in k:
                        new_k = new_k.replace(
                            f'{num_stages}.{num_blocks}.0',
                            f'{num_stages}.blocks.{num_blocks}.spatial_block')
                    elif f'{num_stages}.{num_blocks}.1' in k:
                        new_k = new_k.replace(
                            f'{num_stages}.{num_blocks}.1',
                            f'{num_stages}.blocks.{num_blocks}.channel_block')
            if 'cpe.0' in k:
                new_k = new_k.replace('cpe.0', 'cpe1')
            elif 'cpe.1' in k:
                new_k = new_k.replace('cpe.1', 'cpe2')
            if 'mlp' in k:
                new_k = new_k.replace('mlp.fc1', 'ffn.layers.0.0')
                new_k = new_k.replace('mlp.fc2', 'ffn.layers.1')
            if 'spatial_block.attn' in new_k:
                new_k = new_k.replace('spatial_block.attn',
                                      'spatial_block.attn.w_msa')
        elif k.startswith('norms'):
            new_k = k.replace('norms', 'norm3')
        elif k.startswith('head'):
            new_k = k.replace('head', 'head.fc')
        else:
            new_k = k

        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained davit '
        'models to mmpretrain style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_davit(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

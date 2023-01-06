# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_gpvit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('head'):
            new_k = k
            new_ckpt[k] = new_v
            continue
        elif k.startswith('backbone.patch_embed'):
            new_k = k
            new_k = new_k.replace('stem.0', 'stem.conv')
            new_k = new_k.replace('stem.1', 'stem.bn')
            new_k = new_k.replace('patch_embed.convs.0',
                                  'patch_embed.convs.0.conv')
            new_k = new_k.replace('patch_embed.convs.1',
                                  'patch_embed.convs.0.bn')
            new_k = new_k.replace('patch_embed.convs.3',
                                  'patch_embed.convs.1.conv')
            new_k = new_k.replace('patch_embed.convs.4',
                                  'patch_embed.convs.1.bn')
        elif k.startswith('backbone.layers'):
            new_k = k
            if 'un_group_layer.norm_query' in new_k:
                new_k = new_k.replace('group_layer.norm_query',
                                      'group_layer.norm_x')
            elif 'un_group_layer.norm_key' in new_k:
                new_k = new_k.replace('group_layer.norm_key',
                                      'group_layer.norm_group_token')
            elif 'group_layer.norm_query' in new_k:
                new_k = new_k.replace('group_layer.norm_query',
                                      'group_layer.norm_group_token')
            elif 'group_layer.norm_key' in new_k:
                new_k = new_k.replace('group_layer.norm_key',
                                      'group_layer.norm_x')
            elif 'mixer' in new_k:
                if 'patch_mixer.0' in new_k:
                    new_k = new_k.replace('patch_mixer.0',
                                          'token_mix.layers.0.0')
                elif 'patch_mixer.3' in new_k:
                    new_k = new_k.replace('patch_mixer.3',
                                          'token_mix.layers.1')
                elif 'channel_mixer.0' in new_k:
                    new_k = new_k.replace('channel_mixer.0',
                                          'channel_mix.layers.0.0')
                elif 'channel_mixer.3' in new_k:
                    new_k = new_k.replace('channel_mixer.3',
                                          'channel_mix.layers.1')
                elif 'norm' in new_k:
                    new_k = new_k.replace('norm', 'ln')
            elif 'dwconv.0' in new_k:
                new_k = new_k.replace('dwconv.0', 'dwconv.conv')
            elif 'dwconv.1' in new_k:
                new_k = new_k.replace('dwconv.1', 'dwconv.bn')
        elif k.startswith('backbone.ln1'):
            new_k = k.replace('backbone.ln1', 'backbone.lastnorm')
        else:
            new_k = k

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

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_gpvit(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

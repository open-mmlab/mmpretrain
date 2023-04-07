# Copyright (c) OpenMMLab. All rights reserved.
"""convert the weights of efficientformerv2 in
official(https://github.com/snap-research/EfficientFormer) to mmcls format."""
import argparse
import os.path as osp

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_from_efficientformerv2_timm(param):
    # main change_key
    new_key = dict()
    for name in list(param.keys()):
        data = param[name]
        if 'patch_embed' in name:
            name = 'backbone.' + name
            if '0' in name:
                name = name.replace('0', '0.conv')
            elif '1' in name:
                name = name.replace('1', '0.bn')
            elif '3' in name:
                name = name.replace('3', '1.conv')
            elif '4' in name:
                name = name.replace('4', '1.bn')
        elif 'network' not in name:
            if 'norm' in name:
                name = name.replace('norm', 'backbone.norm6')
            elif 'head' in name:
                name = 'head.' + name
        else:
            name = 'backbone.' + name
            if 'network.1' in name or 'network.3' in name:
                name = name.replace('proj', 'proj.conv')
                name = name.replace('norm', 'proj.bn')
            elif 'network.5' in name:
                if 'conv' in name:
                    name = name.replace('conv', 'conv.conv')
                elif 'bn' in name:
                    name = name.replace('bn', 'conv.bn')
                elif 'attn.proj.1' in name:
                    name = name.replace('proj.1', 'proj.conv')
                elif 'attn.proj.2' in name:
                    name = name.replace('proj.2', 'proj.bn')
                elif '0' in name:
                    name = name.replace('0', 'conv')
                elif '1' in name:
                    name = name.replace('1', 'bn')
            elif 'token_mixer' in name:
                if 'proj.1' in name:
                    name = name.replace('proj.1', 'proj.conv')
                elif 'proj.2' in name:
                    name = name.replace('proj.2', 'proj.bn')
                elif '.0.' in name:
                    name = name.replace('0', 'conv')
                elif '.1.' in name:
                    name = name.replace('1', 'bn')
            else:
                if 'layer_scale_1' in name:
                    name = name.replace('layer_scale_1', 'ls1.weight')
                    data = data.flatten(0)
                elif 'layer_scale_2' in name:
                    name = name.replace('layer_scale_2', 'ls2.weight')
                    data = data.flatten(0)
                elif 'fc1' in name:
                    name = name.replace('fc1', 'fc1.conv')
                elif 'fc2' in name:
                    name = name.replace('fc2', 'fc2.conv')
                elif 'mid_norm' in name:
                    name = name.replace('mid_norm', 'mid.bn')
                elif 'mid' in name:
                    name = name.replace('mid', 'mid.conv')
                elif 'norm1' in name:
                    name = name.replace('norm1', 'fc1.bn')
                elif 'norm2' in name:
                    name = name.replace('norm2', 'fc2.bn')
                elif 'fc1' in name:
                    name = name.replace('fc1', 'fc1.conv')
        new_key[name] = data

    return new_key


def main():
    parser = argparse.ArgumentParser(
        description='Convert pretrained efficientformerv2 '
        'models in official to mmcls style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = convert_from_efficientformerv2_timm(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

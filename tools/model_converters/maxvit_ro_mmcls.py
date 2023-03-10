# Copyright (c) OpenMMLab. All rights reserved.
"""convert the weights of maxcit in
timm(https://github.com/huggingface/pytorch-image-models) to mmcls format."""
import argparse
import os.path as osp

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_from_maxvit_timm(param):
    # main change_key
    new_key = dict()
    if 'stages.2.blocks.12.conv.pre_norm.weight' in param.keys():
        depths = (2, 6, 14, 2)
    else:
        depths = (2, 2, 5, 2)

    new_depth = [0] + [sum(depths[:i + 1]) for i in range(len(depths))]
    new_depth = [list(range(new_depth[i], new_depth[i + 1])) for i in range(len(new_depth) - 1)]

    for name in list(param.keys()):
        data = param[name]
        if 'stem' in name:
            name = 'backbone.' + name
            if 'conv1' in name:
                name = name.replace('conv1', 'conv1.conv')
            elif 'norm1' in name:
                name = name.replace('norm1', 'conv1.bn')
            elif 'conv2' in name:
                name = name.replace('conv2', 'conv2.conv')
            new_key[name] = data
        elif 'head' in name:
            if 'norm' in name:
                name = name.replace('head.norm', 'backbone.gap_neck.1')
            if 'pre_logits.fc' in name:
                name = name.replace('head.pre_logits.fc', 'backbone.gap_neck.3')
            new_key[name] = data
        elif 'stages' in name:
            replace_str = name[:20]
            name_s_lst = replace_str.split('.')
            replace_str = '.'.join(name_s_lst[:4])
            new_num = new_depth[int(name_s_lst[1])][int(name_s_lst[3])]

            name = name.replace(replace_str, 'backbone.layers.' + str(new_num))
            if 'conv.shortcut' in name:
                name = name.replace('conv.shortcut.expand', 'mbconv.shortcut.1')
            elif 'conv.pre_norm' in name:
                name = name.replace('conv.pre_norm', 'mbconv.pre_norm')
            elif 'attn_block' in name:
                name = name.replace('attn_block', 'wind_attn_block')
                name = name.replace('mlp.fc1', 'ffn.layers.0.0')
                name = name.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn_grid' in name:
                name = name.replace('attn_grid', 'grid_attn_block')
                name = name.replace('mlp.fc1', 'ffn.layers.0.0')
                name = name.replace('mlp.fc2', 'ffn.layers.1')
            elif 'conv.conv1_1x1' in name:
                name = name.replace('conv.conv1_1x1', 'mbconv.layers.0.conv')
            elif 'conv.norm1' in name:
                name = name.replace('conv.norm1', 'mbconv.layers.0.bn')
            elif 'conv.conv2_kxk' in name:
                name = name.replace('conv.conv2_kxk', 'mbconv.layers.1.conv')
            elif 'conv.norm2' in name:
                name = name.replace('conv.norm2', 'mbconv.layers.1.bn')
            elif 'conv.se.fc1' in name:
                name = name.replace('conv.se.fc1', 'mbconv.layers.2.conv1.conv')
            elif 'conv.se.fc2' in name:
                name = name.replace('conv.se.fc2', 'mbconv.layers.2.conv2.conv')
            elif 'conv.conv3_1x1' in name:
                name = name.replace('conv.conv3_1x1', 'mbconv.layers.3.conv')

            new_key[name] = data
    return new_key


def main():
    parser = argparse.ArgumentParser(
        description='Convert pretrained maxvit '
        'models in timm to mmcls style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_from_maxvit_timm(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

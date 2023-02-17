# Copyright (c) OpenMMLab. All rights reserved.
"""convert the weights of efficientnetv2 in
timm(https://github.com/rwightman/pytorch-image-models) to mmpretrain
format."""
import argparse
import os.path as osp

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_from_efficientnetv2_timm(param):
    # main change_key
    param_lst = list(param.keys())
    op = str(int(param_lst[-9][7]) + 2)
    new_key = dict()
    for name in param_lst:
        data = param[name]
        if 'blocks' not in name:
            if 'conv_stem' in name:
                name = name.replace('conv_stem', 'backbone.layers.0.conv')
            if 'bn1' in name:
                name = name.replace('bn1', 'backbone.layers.0.bn')
            if 'conv_head' in name:
                # if efficientnet-v2_s/base/b1/b2/b3，op = 7，
                # if for m/l/xl , op = 8
                name = name.replace('conv_head', f'backbone.layers.{op}.conv')
            if 'bn2' in name:
                name = name.replace('bn2', f'backbone.layers.{op}.bn')
            if 'classifier' in name:
                name = name.replace('classifier', 'head.fc')
        else:
            operator = int(name[7])
            if operator == 0:
                name = name[:7] + str(operator + 1) + name[8:]
                name = name.replace('blocks', 'backbone.layers')
                if 'conv' in name:
                    name = name.replace('conv', 'conv')
                if 'bn1' in name:
                    name = name.replace('bn1', 'bn')
            elif operator < 3:
                name = name[:7] + str(operator + 1) + name[8:]
                name = name.replace('blocks', 'backbone.layers')
                if 'conv_exp' in name:
                    name = name.replace('conv_exp', 'conv1.conv')
                if 'conv_pwl' in name:
                    name = name.replace('conv_pwl', 'conv2.conv')
                if 'bn1' in name:
                    name = name.replace('bn1', 'conv1.bn')
                if 'bn2' in name:
                    name = name.replace('bn2', 'conv2.bn')
            else:
                name = name[:7] + str(operator + 1) + name[8:]
                name = name.replace('blocks', 'backbone.layers')
                if 'conv_pwl' in name:
                    name = name.replace('conv_pwl', 'linear_conv.conv')
                if 'conv_pw' in name:
                    name = name.replace('conv_pw', 'expand_conv.conv')
                if 'conv_dw' in name:
                    name = name.replace('conv_dw', 'depthwise_conv.conv')
                if 'bn1' in name:
                    name = name.replace('bn1', 'expand_conv.bn')
                if 'bn2' in name:
                    name = name.replace('bn2', 'depthwise_conv.bn')
                if 'bn3' in name:
                    name = name.replace('bn3', 'linear_conv.bn')
                if 'se.conv_reduce' in name:
                    name = name.replace('se.conv_reduce', 'se.conv1.conv')
                if 'se.conv_expand' in name:
                    name = name.replace('se.conv_expand', 'se.conv2.conv')
        new_key[name] = data
    return new_key


def main():
    parser = argparse.ArgumentParser(
        description='Convert pretrained efficientnetv2 '
        'models in timm to mmpretrain style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_from_efficientnetv2_timm(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_twins(args, ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('linear'):
            new_k = k.replace('linear.', 'head.fc.')
            new_ckpt[new_k] = new_v
            continue
        elif k.startswith('stage'):
            if 'rbr_scale' in k:
                new_k = k.replace('rbr_scale.', 'branch_1x1.')
                new_k = new_k.replace('bn.', 'norm.')
            elif 'rbr_conv' in k:
                new_k = k.replace('rbr_conv.', 'branch_3x3_list.')
                new_k = new_k.replace('bn.', 'norm.')
            elif "rbr_skip" in k:
                new_k = k.replace('rbr_skip.', 'branch_norm.')
                new_k = new_k.replace('bn.', 'norm.')
            elif ".se." in k:
                new_k = k.replace('reduce.', 'conv1.conv.')
                new_k = new_k.replace('expand.', 'conv2.conv.')
            else:
                new_k = k
        new_k = 'backbone.' + new_k
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMClassification style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_twins(args, state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
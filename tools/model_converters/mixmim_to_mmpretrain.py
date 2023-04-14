# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def correct_unfold_reduction_order(x: torch.Tensor):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, 4, in_channel // 4)
    x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
    return x


def correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(4, in_channel // 4)
    x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
    return x


def convert_mixmim(ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v

        if k.startswith('patch_embed'):
            new_k = k.replace('proj', 'projection')

        elif k.startswith('layers'):
            if 'norm1' in k:
                new_k = k.replace('norm1', 'ln1')
            elif 'norm2' in k:
                new_k = k.replace('norm2', 'ln2')
            elif 'mlp.fc1' in k:
                new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = k.replace('mlp.fc2', 'ffn.layers.1')
            else:
                new_k = k

        elif k.startswith('norm') or k.startswith('absolute_pos_embed'):
            new_k = k

        elif k.startswith('head'):
            new_k = k.replace('head.', 'head.fc.')

        else:
            raise ValueError

        # print(new_k)
        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k

        if 'downsample' in new_k:
            print('Covert {} in PatchMerging from timm to mmcv format!'.format(
                new_k))

            if 'reduction' in new_k:
                new_v = correct_unfold_reduction_order(new_v)
            elif 'norm' in new_k:
                new_v = correct_unfold_norm_order(new_v)

        new_ckpt[new_k] = new_v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained mixmim '
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

    weight = convert_mixmim(state_dict)
    # weight = convert_official_mixmim(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

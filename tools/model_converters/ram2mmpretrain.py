# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict
from copy import deepcopy

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_swin(ckpt):
    new_ckpt = OrderedDict()
    convert_mapping = dict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if 'attn_mask' in k:
            continue
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        elif k.startswith('norm'):
            new_v = v
            new_k = k.replace('norm', 'norm3')
        else:
            new_v = v
            new_k = k

        new_ckpt[new_k] = new_v
        convert_mapping[k] = new_k

    return new_ckpt, convert_mapping


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained RAM models to'
        'MMPretrain style.')
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

    visual_ckpt = OrderedDict()
    for key in state_dict:
        if key.startswith('visual_encoder.'):
            new_key = key.replace('visual_encoder.', '')
            visual_ckpt[new_key] = state_dict[key]

    new_visual_ckpt, convert_mapping = convert_swin(visual_ckpt)
    new_ckpt = deepcopy(state_dict)
    for key in state_dict:
        if key.startswith('visual_encoder.'):
            if 'attn_mask' in key:
                del new_ckpt[key]
                continue
            del new_ckpt[key]
            old_key = key.replace('visual_encoder.', '')
            new_ckpt[key.replace(old_key,
                                 convert_mapping[old_key])] = deepcopy(
                                     new_visual_ckpt[key.replace(
                                         old_key,
                                         convert_mapping[old_key]).replace(
                                             'visual_encoder.', '')])

    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(new_ckpt, args.dst)


if __name__ == '__main__':
    main()

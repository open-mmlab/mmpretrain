# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_clip(ckpt):
    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('visual.conv1'):
            new_k = k.replace('conv1', 'patch_embed.projection')
        elif k.startswith('visual.positional_embedding'):
            new_k = k.replace('positional_embedding', 'pos_embed')
            new_v = v.unsqueeze(dim=0)
        elif k.startswith('visual.class_embedding'):
            new_k = k.replace('class_embedding', 'cls_token')
            new_v = v.unsqueeze(dim=0).unsqueeze(dim=0)
        elif k.startswith('visual.ln_pre'):
            new_k = k.replace('ln_pre', 'pre_norm')
        elif k.startswith('visual.transformer.resblocks'):
            new_k = k.replace('transformer.resblocks', 'layers')
            if 'ln_1' in k:
                new_k = new_k.replace('ln_1', 'ln1')
            elif 'ln_2' in k:
                new_k = new_k.replace('ln_2', 'ln2')
            elif 'mlp.c_fc' in k:
                new_k = new_k.replace('mlp.c_fc', 'ffn.layers.0.0')
            elif 'mlp.c_proj' in k:
                new_k = new_k.replace('mlp.c_proj', 'ffn.layers.1')
            elif 'attn.in_proj_weight' in k:
                new_k = new_k.replace('in_proj_weight', 'qkv.weight')
            elif 'attn.in_proj_bias' in k:
                new_k = new_k.replace('in_proj_bias', 'qkv.bias')
            elif 'attn.out_proj' in k:
                new_k = new_k.replace('out_proj', 'proj')
        elif k.startswith('visual.ln_post'):
            new_k = k.replace('ln_post', 'ln1')
        elif k.startswith('visual.proj'):
            new_k = k.replace('visual.proj', 'visual_proj.proj')
        else:
            new_k = k

        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained clip '
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

    weight = convert_clip(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

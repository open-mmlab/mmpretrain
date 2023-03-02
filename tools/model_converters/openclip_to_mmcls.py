# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch

try:
    import open_clip
except ImportError:
    raise ImportError(
        'Failed to import open_clip. Please run "pip install open_clip_torch".'
    )


def convert_openclip(ckpt):
    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        if 'visual' not in k:
            continue
        k = k.replace('visual.', '')
        k = k.replace('transformer.', '')
        new_v = v
        if k.startswith('proj'):
            new_k = 'head.fc.weight'
            new_ckpt[new_k] = new_v.transpose(0, 1)
            new_ckpt['head.fc.bias'] = torch.zeros(
                new_v.transpose(0, 1).shape[0])
            continue
        elif k.startswith('conv1'):
            new_k = k.replace('conv1', 'patch_embed.projection')
        elif k.startswith('ln_pre'):
            new_k = k.replace('ln_pre.', 'pre_norm.')
        elif k.startswith('resblocks'):
            new_k = k.replace('resblocks.', 'layers.')
            if 'ln_1' in k:
                new_k = new_k.replace('ln_1', 'ln1')
            elif 'ln_2' in k:
                new_k = new_k.replace('ln_2', 'ln2')
            elif 'in_proj_weight' in k:
                new_k = new_k.replace('in_proj_weight', 'qkv.weight')
            elif 'in_proj_bias' in k:
                new_k = new_k.replace('in_proj_bias', 'qkv.bias')
            elif 'out_proj' in k:
                new_k = new_k.replace('out_proj', 'proj')
            elif 'mlp.c_fc' in k:
                new_k = new_k.replace('mlp.c_fc', 'ffn.layers.0.0')
            elif 'mlp.c_proj' in k:
                new_k = new_k.replace('mlp.c_proj', 'ffn.layers.1')
        elif k.startswith('class_embedding'):
            new_v = v[None, None, :]
            new_k = k.replace('class_embedding', 'cls_token')
        elif k.startswith('positional_embedding'):
            new_v = v[None, ...]
            new_k = k.replace('positional_embedding', 'pos_embed')
        elif k.startswith('ln_post'):
            new_k = k.replace('ln_post', 'ln1')
        else:
            new_k = k

        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained models to mmcls style.')
    parser.add_argument('model_name', help='open_clip model name')
    parser.add_argument('pretrained', help='open_clip pretrained name')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    # load torch.jit model weights
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=args.model_name, pretrained=args.pretrained)
    state_dict = model.state_dict()

    weight = convert_openclip(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    print(weight['backbone.pos_embed'].shape)
    torch.save(dict(state_dict=weight), args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

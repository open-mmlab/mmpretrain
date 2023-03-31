# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_eva02(ckpt):

    new_ckpt = OrderedDict()
    qkv_proj = {}
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
    qkv_bias = {}

    banned = {
        'mask_token',
        'lm_head.weight',
        'lm_head.bias',
        'norm.weight',
        'norm.bias',
=======

    banned = {
        'mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight',
        'norm.bias'
>>>>>>> feat: add eva02 backbone
    }

    for k, v in list(ckpt.items()):

        if k in banned:
            continue

        if k.startswith('head'):
            new_k = k.replace('head.', 'head.fc.')
            new_ckpt[new_k] = v
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536

=======
>>>>>>> feat: add eva02 backbone
        else:
            if k.startswith('patch_embed'):
                new_k = k.replace('proj.', 'projection.')

            elif k.startswith('fc_norm') or k.startswith('norm'):
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
                new_k = k.replace('norm.', 'ln1.')
                new_k = k.replace('fc_norm.', 'ln1.')
=======
                new_k = k.replace('norm.', 'final_norm.')
                new_k = k.replace('fc_norm.', 'final_norm.')
>>>>>>> feat: add eva02 backbone

            elif k.startswith('blocks'):
                new_k = k.replace('blocks.', 'layers.')

                if 'mlp' in new_k:
                    if 'w12' in new_k:
                        # For tiny and small version, mlp is implemented with
                        # swiglu in xformers, where w1 and w2 are integrated
                        # into w12.
                        out_dim = v.shape[0]
                        if 'weight' in new_k:
                            # w12.weight
                            v = v.reshape(2, out_dim // 2, -1)
                        else:
                            # w12.bias
                            v = v.reshape(2, out_dim // 2)
                        new_k = new_k.replace('w12.', 'w1.')
                        new_ckpt['backbone.' + new_k] = v[0]
                        new_k = new_k.replace('w1.', 'w2.')
                        new_ckpt['backbone.' + new_k] = v[1]
                        continue

                    if 'ffn_ln' in new_k:
                        new_k = new_k.replace('ffn_ln.', 'norm.')

                elif 'attn' in new_k:
<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536

=======
>>>>>>> feat: add eva02 backbone
                    if 'q_proj.weight' in new_k or \
                            'k_proj.weight' in new_k or \
                            'v_proj.weight' in new_k:
                        # For base and large version, qkv projection is
                        # implemented with three linear layers,
                        s = new_k.split('.')
                        idx = s[1]
                        if idx not in qkv_proj:
                            qkv_proj[idx] = {}
                        qkv_proj[idx][s[-2]] = v
                        continue

<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
                    if 'q_bias' in new_k or 'v_bias' in new_k:
                        # k_bias is 0
                        s = new_k.split('.')
                        idx = s[1]
                        if idx not in qkv_bias:
                            qkv_bias[idx] = {}
                        qkv_bias[idx][s[-1]] = v
                        continue

=======
>>>>>>> feat: add eva02 backbone
            else:
                new_k = k

            new_k = 'backbone.' + new_k
            new_ckpt[new_k] = v

    for idx in qkv_proj:
        q_proj = qkv_proj[idx]['q_proj']
        k_proj = qkv_proj[idx]['k_proj']
        v_proj = qkv_proj[idx]['v_proj']
        weight = torch.cat((q_proj, k_proj, v_proj))
        new_k = f'backbone.layers.{idx}.attn.qkv.weight'
        new_ckpt[new_k] = weight

<<<<<<< 07f4ad5a25e02ece517f676d234ffe2d790d1536
    for idx in qkv_bias:
        q_bias = qkv_bias[idx]['q_bias']
        k_bias = torch.zeros_like(q_bias)
        v_bias = qkv_bias[idx]['v_bias']
        weight = torch.cat((q_bias, k_bias, v_bias))
        new_k = f'backbone.layers.{idx}.attn.qkv.bias'
        new_ckpt[new_k] = weight

=======
>>>>>>> feat: add eva02 backbone
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained eva02 '
        'models to mmpretrain style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint

    weight = convert_eva02(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

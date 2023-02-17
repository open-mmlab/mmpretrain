# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_revvit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('head.projection'):
            new_k = k.replace('head.projection', 'head.fc')
            new_ckpt[new_k] = new_v
            continue
        elif k.startswith('patch_embed'):
            if 'proj.' in k:
                new_k = k.replace('proj.', 'projection.')
            else:
                new_k = k
        elif k.startswith('rev_backbone'):
            new_k = k.replace('rev_backbone.', '')
            if 'F.norm' in k:
                new_k = new_k.replace('F.norm', 'ln1')
            elif 'G.norm' in k:
                new_k = new_k.replace('G.norm', 'ln2')
            elif 'F.attn' in k:
                new_k = new_k.replace('F.attn', 'attn')
            elif 'G.mlp.fc1' in k:
                new_k = new_k.replace('G.mlp.fc1', 'ffn.layers.0.0')
            elif 'G.mlp.fc2' in k:
                new_k = new_k.replace('G.mlp.fc2', 'ffn.layers.1')
        elif k.startswith('norm'):
            new_k = k.replace('norm', 'ln1')
        else:
            new_k = k

        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k
        new_ckpt[new_k] = new_v

    tmp_weight_dir = []
    tmp_bias_dir = []
    final_ckpt = OrderedDict()
    for k, v in list(new_ckpt.items()):
        if 'attn.q.weight' in k:
            tmp_weight_dir.append(v)
        elif 'attn.k.weight' in k:
            tmp_weight_dir.append(v)
        elif 'attn.v.weight' in k:
            tmp_weight_dir.append(v)
            new_k = k.replace('attn.v.weight', 'attn.qkv.weight')
            final_ckpt[new_k] = torch.cat(tmp_weight_dir, dim=0)
            tmp_weight_dir = []
        elif 'attn.q.bias' in k:
            tmp_bias_dir.append(v)
        elif 'attn.k.bias' in k:
            tmp_bias_dir.append(v)
        elif 'attn.v.bias' in k:
            tmp_bias_dir.append(v)
            new_k = k.replace('attn.v.bias', 'attn.qkv.bias')
            final_ckpt[new_k] = torch.cat(tmp_bias_dir, dim=0)
            tmp_bias_dir = []
        else:
            final_ckpt[k] = v

        # add pos embed for cls token
        if k == 'backbone.pos_embed':
            v = torch.cat([torch.ones_like(v).mean(dim=1, keepdim=True), v],
                          dim=1)
            final_ckpt[k] = v

    return final_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained revvit'
        ' models to mmpretrain style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    weight = convert_revvit(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

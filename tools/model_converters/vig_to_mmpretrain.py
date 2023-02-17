# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import re
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_vig(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        new_key = k
        new_value = v
        if 'pos_embed' in new_key:
            new_key = new_key.replace('pos_embed', 'backbone.pos_embed')
        elif 'stem' in new_key:
            new_key = new_key.replace('stem.convs', 'backbone.stem')
        elif 'backbone' in new_key:
            new_key = new_key.replace('backbone', 'backbone.blocks')
        elif 'prediction.0' in new_key:
            new_key = new_key.replace('prediction.0', 'head.fc1')
            new_value = v.squeeze(-1).squeeze(-1)
        elif 'prediction.1' in new_key:
            new_key = new_key.replace('prediction.1', 'head.bn')
        elif 'prediction.4' in new_key:
            new_key = new_key.replace('prediction.4', 'head.fc2')
            new_value = v.squeeze(-1).squeeze(-1)
        new_ckpt[new_key] = new_value
    return new_ckpt


def convert_pvig(ckpt):
    new_ckpt = OrderedDict()

    stage_idx = 0
    stage_blocks = 0
    for k, v in ckpt.items():
        new_key: str = k
        new_value = v
        if 'pos_embed' in new_key:
            new_key = new_key.replace('pos_embed', 'backbone.pos_embed')
        elif 'stem' in new_key:
            new_key = new_key.replace('stem.convs', 'backbone.stem')
        elif re.match(r'^backbone\.\d+\.conv', new_key) is not None:
            if new_key.endswith('0.weight'):
                stage_idx += 1
            stage_blocks = int(new_key.split('.')[1])
            other = new_key.split('.', maxsplit=3)[-1]
            new_key = f'backbone.stages.{stage_idx}.0.' + other
        elif 'backbone' in new_key:
            block_idx = int(new_key.split('.')[1]) - stage_blocks
            other = new_key.split('.', maxsplit=2)[-1]
            new_key = f'backbone.stages.{stage_idx}.{block_idx}.' + other
        elif 'prediction.0' in new_key:
            new_key = new_key.replace('prediction.0', 'head.fc1')
            new_value = v.squeeze(-1).squeeze(-1)
        elif 'prediction.1' in new_key:
            new_key = new_key.replace('prediction.1', 'head.bn')
        elif 'prediction.4' in new_key:
            new_key = new_key.replace('prediction.4', 'head.fc2')
            new_value = v.squeeze(-1).squeeze(-1)
        new_ckpt[new_key] = new_value
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained vig models to '
        'mmpretrain style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    if 'backbone.2.conv.0.weight' in state_dict:
        weight = convert_pvig(state_dict)
    else:
        weight = convert_vig(state_dict)

    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

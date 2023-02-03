# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_densenet(ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        new_k = k.replace('module.densenet121.features.',
                          '').replace('module.densenet121.classifier.0.',
                                      'head.fc.')
        if new_k.startswith('head'):
            new_ckpt[new_k] = new_v
            continue
        elif new_k.startswith('conv0'):
            new_k = new_k.replace('conv0', 'stem.0')
        elif new_k.startswith('norm0'):
            new_k = new_k.replace('norm0', 'stem.1')
        elif new_k.startswith('denseblock'):
            new_k = new_k.replace('conv.', 'conv').replace('norm.', 'norm')
            for i in range(10):
                if new_k.startswith(f'denseblock{i+1}.'):
                    new_k = new_k.replace(f'denseblock{i+1}.', f'stages.{i}.')
                    break
            for j in range(30):
                if f'denselayer{j+1}.' in new_k:
                    new_k = new_k.replace(f'denselayer{j+1}.', f'block.{j}.')
                    break
        elif new_k.startswith('transition'):
            for i in range(10):
                if new_k.startswith(f'transition{i+1}.'):
                    new_k = new_k.replace(f'transition{i+1}.',
                                          f'transitions.{i}.')
                    break
        elif new_k.startswith('norm5'):
            new_k = new_k.replace('norm5', 'transitions.3.0')

        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained chexnet models to mmcls style.'
    )
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_densenet(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(dict(state_dict=weight), args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()

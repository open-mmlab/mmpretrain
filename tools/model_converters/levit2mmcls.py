# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import load_checkpoint

from mmcls.models.backbones.levit import get_LeViT_model


def convert_levit(args, ckpt):

    model = get_LeViT_model(args.type)
    origin = ckpt
    new = model.state_dict()
    keys = []
    keys1 = []

    for key, _ in origin.items():
        keys.append(key)
    for key, _ in new.items():
        keys1.append(key)

    change_dict = {}
    for i in range(len(keys)):
        change_dict[keys1[i]] = keys[i]

    with torch.no_grad():
        for name, param in new.items():
            param.copy_(origin[change_dict[name]])
    return new


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMClassification style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    parser.add_argument('type', default='LeViT-256', help='模型的种类(128S、128等)')
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location='cpu')
    checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_levit(args, state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()

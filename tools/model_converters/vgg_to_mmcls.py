# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import OrderedDict

import torch


def get_layer_maps(layer_num, with_bn):
    layer_maps = {'conv': {}, 'bn': {}}
    if with_bn:
        if layer_num == 11:
            layer_idxs = [0, 4, 8, 11, 15, 18, 22, 25]
        elif layer_num == 13:
            layer_idxs = [0, 3, 7, 10, 14, 17, 21, 24, 28, 31]
        elif layer_num == 16:
            layer_idxs = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        elif layer_num == 19:
            layer_idxs = [
                0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49
            ]
        else:
            raise ValueError(f'Invalid number of layers: {layer_num}')
        for i, layer_idx in enumerate(layer_idxs):
            if i == 0:
                new_layer_idx = layer_idx
            else:
                new_layer_idx += int((layer_idx - layer_idxs[i - 1]) / 2)
            layer_maps['conv'][layer_idx] = new_layer_idx
            layer_maps['bn'][layer_idx + 1] = new_layer_idx
    else:
        if layer_num == 11:
            layer_idxs = [0, 3, 6, 8, 11, 13, 16, 18]
            new_layer_idxs = [0, 2, 4, 5, 7, 8, 10, 11]
        elif layer_num == 13:
            layer_idxs = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22]
            new_layer_idxs = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]
        elif layer_num == 16:
            layer_idxs = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
            new_layer_idxs = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
        elif layer_num == 19:
            layer_idxs = [
                0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34
            ]
            new_layer_idxs = [
                0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19
            ]
        else:
            raise ValueError(f'Invalid number of layers: {layer_num}')

        layer_maps['conv'] = {
            layer_idx: new_layer_idx
            for layer_idx, new_layer_idx in zip(layer_idxs, new_layer_idxs)
        }

    return layer_maps


def convert(src, dst, layer_num, with_bn=False):
    """Convert keys in torchvision pretrained VGG models to mmcls style."""

    # load pytorch model
    assert os.path.isfile(src), f'no checkpoint found at {src}'
    blobs = torch.load(src, map_location='cpu')

    # convert to pytorch style
    state_dict = OrderedDict()

    layer_maps = get_layer_maps(layer_num, with_bn)

    prefix = 'backbone'
    delimiter = '.'
    for key, weight in blobs.items():
        if 'features' in key:
            module, layer_idx, weight_type = key.split(delimiter)
            new_key = delimiter.join([prefix, key])
            layer_idx = int(layer_idx)
            for layer_key, maps in layer_maps.items():
                if layer_idx in maps:
                    new_layer_idx = maps[layer_idx]
                    new_key = delimiter.join([
                        prefix, 'features',
                        str(new_layer_idx), layer_key, weight_type
                    ])
            state_dict[new_key] = weight
            print(f'Convert {key} to {new_key}')
        elif 'classifier' in key:
            new_key = delimiter.join([prefix, key])
            state_dict[new_key] = weight
            print(f'Convert {key} to {new_key}')
        else:
            state_dict[key] = weight

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src torchvision model path')
    parser.add_argument('dst', help='save path')
    parser.add_argument(
        '--bn', action='store_true', help='whether original vgg has BN')
    parser.add_argument(
        '--layer-num',
        type=int,
        choices=[11, 13, 16, 19],
        default=11,
        help='number of VGG layers')
    args = parser.parse_args()
    convert(args.src, args.dst, layer_num=args.layer_num, with_bn=args.bn)


if __name__ == '__main__':
    main()

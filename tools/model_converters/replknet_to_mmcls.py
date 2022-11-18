# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def convert(src, dst):
    print('Converting...')
    blobs = torch.load(src, map_location='cpu')
    converted_state_dict = OrderedDict()

    for key in blobs:
        splited_key = key.split('.')
        print(splited_key)
        splited_key = [
            'backbone.stem' if i[:4] == 'stem' else i for i in splited_key
        ]
        splited_key = [
            'backbone.stages' if i[:6] == 'stages' else i for i in splited_key
        ]
        splited_key = [
            'backbone.transitions' if i[:11] == 'transitions' else i
            for i in splited_key
        ]
        splited_key = [
            'backbone.stages.3.norm' if i[:4] == 'norm' else i
            for i in splited_key
        ]
        splited_key = [
            'head.fc' if i[:4] == 'head' else i for i in splited_key
        ]

        new_key = '.'.join(splited_key)
        converted_state_dict[new_key] = blobs[key]

    torch.save(converted_state_dict, dst)
    print('Done!')


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    dst = Path(args.dst)
    if dst.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)

    convert(args.src, args.dst)


if __name__ == '__main__':
    main()

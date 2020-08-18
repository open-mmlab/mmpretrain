import argparse
import os
from collections import OrderedDict

import torch


def convert(src, dst):
    """Convert keys in torchvision pretrained VGG models to mmcls
    style."""

    # load pytorch model
    assert os.path.isfile(src), f'no checkpoint found at {src}'
    blobs = torch.load(src, map_location='cpu')

    # convert to pytorch style
    state_dict = OrderedDict()

    prefix = 'backbone.'
    for key, weight in blobs.items():
        if 'features' in key or 'classifier' in key:
            state_dict[prefix + key] = weight
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
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()

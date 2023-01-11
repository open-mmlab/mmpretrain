# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import collections

import torch


def convert(src, dst):
    state = torch.load(src)
    new_state = collections.OrderedDict()
    for key in state.keys():
        new_key = key
        if 'backbone' in new_key:
            new_key = new_key.replace('backbone', 'stage_blocks')
        new_key = 'backbone.' + new_key
        if 'prediction' in new_key:
            new_key = new_key.replace('prediction', 'classifier')
            new_key = new_key.replace('backbone', 'head')
        new_state[new_key] = state[key]
    torch.save(new_state, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('--src', help='src path')
    parser.add_argument('--dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()

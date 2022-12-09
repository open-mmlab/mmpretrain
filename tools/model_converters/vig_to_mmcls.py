# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import collections

import torch


def convert(src, dst):
    state = torch.load(src)
    newstate = collections.OrderedDict()
    for key in state.keys():
        newkey = key
        if 'gconv.' in newkey:
            newkey = newkey.replace('gconv.', '')
        newkey = 'backbone.' + newkey
        newstate[newkey] = state[key]
    torch.save(newstate, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('--src', help='src path')
    parser.add_argument('--dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()

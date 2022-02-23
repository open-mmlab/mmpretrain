import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def convert(src, dst):
    print('Converting...')
    blobs = torch.load(src, map_location='cpu')
    converted_state_dict = OrderedDict()

    for key, value in blobs.items():
        if key.startswith('head.'):
            splited_key = key.split('.')
            splited_key = [
                'head.fc' if i == 'head' else i for i in splited_key
            ]
            new_key = '.'.join(splited_key)
            converted_state_dict[new_key] = value
        else:
            splited_key = key.split('.')
            splited_key = [
                'path_embed' if i == 'conv_embedding' else i
                for i in splited_key
            ]
            splited_key = [
                'downsample_layers' if i == 'embeds' else i
                for i in splited_key
            ]
            splited_key = [
                'final_norm' if i == 'head_norm' else i for i in splited_key
            ]
            splited_key = [
                'conv1.conv' if i == 'fc1' else i for i in splited_key
            ]
            splited_key = [
                'conv2.conv' if i == 'fc2' else i for i in splited_key
            ]
            splited_key = [
                'norm1' if i == 'prebn1' else i for i in splited_key
            ]
            splited_key = [
                'norm2' if i == 'prebn2' else i for i in splited_key
            ]
            new_key = 'backbone.' + '.'.join(splited_key)
            converted_state_dict[new_key] = value
        print(f'{new_key} <--- {key} | Size: {value.size()}')
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
        exit()
    dst.parent.mkdir(parents=True, exist_ok=True)

    convert(args.src, args.dst)


if __name__ == '__main__':
    main()

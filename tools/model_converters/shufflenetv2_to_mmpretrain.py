# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch


def convert_conv1(model_key, model_weight, state_dict, converted_names):
    if model_key.find('conv1.0') >= 0:
        new_key = model_key.replace('conv1.0', 'backbone.conv1.conv')
    else:
        new_key = model_key.replace('conv1.1', 'backbone.conv1.bn')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_conv5(model_key, model_weight, state_dict, converted_names):
    if model_key.find('conv5.0') >= 0:
        new_key = model_key.replace('conv5.0', 'backbone.layers.3.conv')
    else:
        new_key = model_key.replace('conv5.1', 'backbone.layers.3.bn')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_head(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('fc', 'head.fc')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_block(model_key, model_weight, state_dict, converted_names):
    split_keys = model_key.split('.')
    layer, block, branch = split_keys[:3]
    layer_id = int(layer[-1]) - 2
    new_key = model_key.replace(layer, f'backbone.layers.{layer_id}')

    if branch == 'branch1':
        if new_key.find('branch1.0') >= 0:
            new_key = new_key.replace('branch1.0', 'branch1.0.conv')
        elif new_key.find('branch1.1') >= 0:
            new_key = new_key.replace('branch1.1', 'branch1.0.bn')
        elif new_key.find('branch1.2') >= 0:
            new_key = new_key.replace('branch1.2', 'branch1.1.conv')
        elif new_key.find('branch1.3') >= 0:
            new_key = new_key.replace('branch1.3', 'branch1.1.bn')
    elif branch == 'branch2':

        if new_key.find('branch2.0') >= 0:
            new_key = new_key.replace('branch2.0', 'branch2.0.conv')
        elif new_key.find('branch2.1') >= 0:
            new_key = new_key.replace('branch2.1', 'branch2.0.bn')
        elif new_key.find('branch2.3') >= 0:
            new_key = new_key.replace('branch2.3', 'branch2.1.conv')
        elif new_key.find('branch2.4') >= 0:
            new_key = new_key.replace('branch2.4', 'branch2.1.bn')
        elif new_key.find('branch2.5') >= 0:
            new_key = new_key.replace('branch2.5', 'branch2.2.conv')
        elif new_key.find('branch2.6') >= 0:
            new_key = new_key.replace('branch2.6', 'branch2.2.bn')
        else:
            raise ValueError(f'Unsupported conversion of key {model_key}')
    else:
        raise ValueError(f'Unsupported conversion of key {model_key}')
    print(f'Convert {model_key} to {new_key}')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)


def convert(src, dst):
    """Convert keys in torchvision pretrained ShuffleNetV2 models to mmpretrain
    style."""

    # load pytorch model
    blobs = torch.load(src, map_location='cpu')

    # convert to pytorch style
    state_dict = OrderedDict()
    converted_names = set()

    for key, weight in blobs.items():
        if 'conv1' in key:
            convert_conv1(key, weight, state_dict, converted_names)
        elif 'fc' in key:
            convert_head(key, weight, state_dict, converted_names)
        elif key.startswith('s'):
            convert_block(key, weight, state_dict, converted_names)
        elif 'conv5' in key:
            convert_conv5(key, weight, state_dict, converted_names)

    # check if all layers are converted
    for key in blobs:
        if key not in converted_names:
            print(f'not converted: {key}')
    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()

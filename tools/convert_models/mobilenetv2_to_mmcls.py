# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch


def convert_conv1(model_key, model_weight, state_dict, converted_names):
    if model_key.find('features.0.0') >= 0:
        new_key = model_key.replace('features.0.0', 'backbone.conv1.conv')
    else:
        new_key = model_key.replace('features.0.1', 'backbone.conv1.bn')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_conv5(model_key, model_weight, state_dict, converted_names):
    if model_key.find('features.18.0') >= 0:
        new_key = model_key.replace('features.18.0', 'backbone.conv2.conv')
    else:
        new_key = model_key.replace('features.18.1', 'backbone.conv2.bn')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_head(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('classifier.1', 'head.fc')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_block(model_key, model_weight, state_dict, converted_names):
    split_keys = model_key.split('.')
    layer_id = int(split_keys[1])
    new_layer_id = 0
    sub_id = 0
    if layer_id == 1:
        new_layer_id = 1
        sub_id = 0
    elif layer_id in range(2, 4):
        new_layer_id = 2
        sub_id = layer_id - 2
    elif layer_id in range(4, 7):
        new_layer_id = 3
        sub_id = layer_id - 4
    elif layer_id in range(7, 11):
        new_layer_id = 4
        sub_id = layer_id - 7
    elif layer_id in range(11, 14):
        new_layer_id = 5
        sub_id = layer_id - 11
    elif layer_id in range(14, 17):
        new_layer_id = 6
        sub_id = layer_id - 14
    elif layer_id == 17:
        new_layer_id = 7
        sub_id = 0

    new_key = model_key.replace(f'features.{layer_id}',
                                f'backbone.layer{new_layer_id}.{sub_id}')
    if new_layer_id == 1:
        if new_key.find('conv.0.0') >= 0:
            new_key = new_key.replace('conv.0.0', 'conv.0.conv')
        elif new_key.find('conv.0.1') >= 0:
            new_key = new_key.replace('conv.0.1', 'conv.0.bn')
        elif new_key.find('conv.1') >= 0:
            new_key = new_key.replace('conv.1', 'conv.1.conv')
        elif new_key.find('conv.2') >= 0:
            new_key = new_key.replace('conv.2', 'conv.1.bn')
        else:
            raise ValueError(f'Unsupported conversion of key {model_key}')
    else:
        if new_key.find('conv.0.0') >= 0:
            new_key = new_key.replace('conv.0.0', 'conv.0.conv')
        elif new_key.find('conv.0.1') >= 0:
            new_key = new_key.replace('conv.0.1', 'conv.0.bn')
        elif new_key.find('conv.1.0') >= 0:
            new_key = new_key.replace('conv.1.0', 'conv.1.conv')
        elif new_key.find('conv.1.1') >= 0:
            new_key = new_key.replace('conv.1.1', 'conv.1.bn')
        elif new_key.find('conv.2') >= 0:
            new_key = new_key.replace('conv.2', 'conv.2.conv')
        elif new_key.find('conv.3') >= 0:
            new_key = new_key.replace('conv.3', 'conv.2.bn')
        else:
            raise ValueError(f'Unsupported conversion of key {model_key}')
    print(f'Convert {model_key} to {new_key}')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)


def convert(src, dst):
    """Convert keys in torchvision pretrained MobileNetV2 models to mmcls
    style."""

    # load pytorch model
    blobs = torch.load(src, map_location='cpu')

    # convert to pytorch style
    state_dict = OrderedDict()
    converted_names = set()

    for key, weight in blobs.items():
        if 'features.0' in key:
            convert_conv1(key, weight, state_dict, converted_names)
        elif 'classifier' in key:
            convert_head(key, weight, state_dict, converted_names)
        elif 'features.18' in key:
            convert_conv5(key, weight, state_dict, converted_names)
        else:
            convert_block(key, weight, state_dict, converted_names)

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

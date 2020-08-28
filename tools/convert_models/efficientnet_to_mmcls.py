import argparse
from collections import OrderedDict

import torch


def convert_conv1(model_key, model_weight, state_dict, converted_names):
    if model_key.find('conv1') >= 0:
        new_key = model_key.replace('conv1', 'backbone.layers.0.conv')
    elif model_key.find('bn1') >= 0:
        new_key = model_key.replace('bn1', 'backbone.layers.0.bn')
    else:
        raise ValueError(f'Unsupported conversion of key {model_key}')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_conv_last(model_key, model_weight, state_dict, converted_names):
    if model_key.find('conv_out') >= 0:
        new_key = model_key.replace('conv_out', 'backbone.layers.6.conv')
    elif model_key.find('bn_out') >= 0:
        new_key = model_key.replace('bn_out', 'backbone.layers.6.bn')
    else:
        raise ValueError(f'Unsupported conversion of key {model_key}')
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
    layer, block = split_keys[:2]
    layer_id = int(layer[-1]) + 1
    new_key = model_key.replace(layer, f'backbone.layers.{layer_id}')

    if new_key.find('conv1') >= 0:
        new_key = new_key.replace('conv1', 'conv1.conv')
    elif new_key.find('bn1') >= 0:
        new_key = new_key.replace('bn1', 'conv1.bn')
    elif new_key.find('conv2') >= 0:
        new_key = new_key.replace('conv2', 'conv2.conv')
    elif new_key.find('bn2') >= 0:
        new_key = new_key.replace('bn2', 'conv2.bn')
    elif new_key.find('conv3') >= 0:
        new_key = new_key.replace('conv3', 'conv3.conv')
    elif new_key.find('bn3') >= 0:
        new_key = new_key.replace('bn3', 'conv3.bn')
    elif new_key.find('se.conv.0') >= 0:
        new_key = new_key.replace('se.conv.0', 'se.conv1.conv')
    elif new_key.find('se.conv.2') >= 0:
        new_key = new_key.replace('se.conv.2', 'se.conv2.conv')
    elif new_key.find('branch1.3') >= 0:
        new_key = new_key.replace('branch1.3', 'branch1.1.bn')
    else:
        raise ValueError(f'Unsupported conversion of key {model_key}')
    print(f'Convert {model_key} to {new_key}')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)


def convert(src, dst):
    """Convert keys in torchvision pretrained ShuffleNetV2 models to mmcls
    style."""

    # load pytorch model
    blobs = torch.load(src, map_location='cpu')

    # convert to pytorch style
    state_dict = OrderedDict()
    converted_names = set()

    for key, weight in blobs.items():
        if key.startswith('conv1') or key.startswith('bn1'):
            convert_conv1(key, weight, state_dict, converted_names)
        elif 'fc' in key:
            convert_head(key, weight, state_dict, converted_names)
        elif key.startswith('l'):
            convert_block(key, weight, state_dict, converted_names)
        elif key.find('out'):
            convert_conv_last(key, weight, state_dict, converted_names)

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

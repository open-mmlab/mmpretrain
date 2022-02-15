import argparse
from pathlib import Path

import torch


def convert_weights(weight):
    """Weight Converter.

    Converts the weights from timm to mmcls

    Args:
        weight (dict): weight dict from timm

    Returns: converted weight dict for mmcls
    """
    result = dict()
    result['meta'] = dict()
    temp = dict()
    mapping = {
        'stem': 'patch_embed',
        'proj': 'projection',
        'mlp_tokens.fc1': 'token_mix.layers.0.0',
        'mlp_tokens.fc2': 'token_mix.layers.1',
        'mlp_channels.fc1': 'channel_mix.layers.0.0',
        'mlp_channels.fc2': 'channel_mix.layers.1',
        'norm1': 'ln1',
        'norm2': 'ln2',
        'norm.': 'ln1.',
        'blocks': 'layers'
    }
    for k, v in weight.items():
        for mk, mv in mapping.items():
            if mk in k:
                k = k.replace(mk, mv)
        if k.startswith('head.'):
            temp['head.fc.' + k[5:]] = v
        else:
            temp['backbone.' + k] = v
    result['state_dict'] = temp
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    dst = Path(args.dst)
    if dst.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)

    original_model = torch.load(args.src, map_location='cpu')
    converted_model = convert_weights(original_model)
    torch.save(converted_model, args.dst)

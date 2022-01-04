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
    for k, v in weight.items():
        if k.startswith('head.fc'):
            temp[k] = v
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
    dst.parent.mkdir(parents=True, exist_ok=True)

    original_model = torch.load(args.src, map_location='cpu')['model_state']
    converted_model = convert_weights(original_model)
    torch.save(converted_model, args.dst)

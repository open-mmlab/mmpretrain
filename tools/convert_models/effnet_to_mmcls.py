import argparse
from pathlib import Path

import torch


def convert_weights(weight):
    """Weight Converter.

    Converts the weights from pycls to mmcls

    Args:
        weight (dict): weight dict from pycls

    Returns: converted weight dict for mmcls
    """
    result = dict()
    result['meta'] = dict()
    temp = dict()
    mapping = {
        # 'se.f_ex.0': 'se.conv1.conv',
        # 'se.f_ex.2': 'se.conv2.conv',
        'stem.conv': 'stem.0',
        'stem.bn': 'stem.1'
    }
    for k, v in weight.items():
        for mk, mv in mapping.items():
            if mk in k:
                k = k.replace(mk, mv)
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

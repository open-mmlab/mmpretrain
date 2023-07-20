# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path

import torch


def merge_lora_weight(base_weight, lora_weight):
    """Merge base weight and lora weight.

    Args:
        base_weight (dict): weight dict from LoRAModel.module.
        lora_weight (dict): weight dict from LoRAModel.
    Returns:
        Merged weight.
    """
    temp = dict()
    mapping = dict()

    for lora_key, lora_value in lora_weight['state_dict'].items():
        # backbone.module.layers.0.attn.qkv.lora_up.weight
        lora_prefix = '.'.join(lora_key.split('.')[:-2])
        mapping[lora_prefix] = f'{lora_prefix}.original_layer'
        temp[lora_key] = lora_value

    for base_key, base_value in base_weight.items():
        # backbone.patch_embed.projection.weight
        # add 'module' prefix
        base_split = base_key.split('.')
        base_split.insert(1, 'module')
        base_prefix = '.'.join(base_split[:-1])
        if base_prefix in mapping.keys():
            base_prefix = mapping[base_prefix]
        base_key = '.'.join([base_prefix, base_split[-1]])
        temp[base_key] = base_value

    result = dict()
    result['state_dict'] = temp
    for key in lora_weight.keys():
        if key != 'state_dict':
            result[key] = lora_weight[key]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge LoRA weight')
    parser.add_argument('base_src', help='src detectron base model path')
    parser.add_argument('lora_src', help='src detectron lora model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    dst = Path(args.dst)
    if dst.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)

    base_model = torch.load(args.base_src, map_location='cpu')
    lora_model = torch.load(args.lora_src, map_location='cpu')

    merged_model = merge_lora_weight(base_model, lora_model)
    torch.save(merged_model, args.dst)

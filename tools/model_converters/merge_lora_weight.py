# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path

import torch
from mmengine.config import Config

from mmpretrain.registry import MODELS


@torch.no_grad()
def merge_lora_weight(cfg, lora_weight):
    """Merge base weight and lora weight.

    Args:
        cfg (dict): config for LoRAModel.
        lora_weight (dict): weight dict from LoRAModel.
    Returns:
        Merged weight.
    """
    temp = dict()
    mapping = dict()
    for name, param in lora_weight['state_dict'].items():
        # backbone.module.layers.11.attn.qkv.lora_down.weight
        if '.lora_' in name:
            lora_split = name.split('.')
            prefix = '.'.join(lora_split[:-2])
            if prefix not in mapping:
                mapping[prefix] = dict()
            lora_type = lora_split[-2]
            mapping[prefix][lora_type] = param
        else:
            temp[name] = param

    model = MODELS.build(cfg['model'])
    for name, param in model.named_parameters():
        if name in temp or '.lora_' in name:
            continue
        else:
            name_split = name.split('.')
            prefix = prefix = '.'.join(name_split[:-2])
            if prefix in mapping:
                name_split.pop(-2)
                if name_split[-1] == 'weight':
                    scaling = get_scaling(model, prefix)
                    lora_down = mapping[prefix]['lora_down']
                    lora_up = mapping[prefix]['lora_up']
                    param += lora_up @ lora_down * scaling
            name_split.pop(1)
            name = '.'.join(name_split)
            temp[name] = param

    result = dict()
    result['state_dict'] = temp
    result['meta'] = lora_weight['meta']
    return result


def get_scaling(model, prefix):
    """Get the scaling of target layer.

    Args:
        model (LoRAModel): the LoRAModel.
        prefix (str): the prefix of the layer.
    Returns:
        the scale of the LoRALinear.
    """
    prefix_split = prefix.split('.')
    for i in prefix_split:
        model = getattr(model, i)
    return model.scaling


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge LoRA weight')
    parser.add_argument('cfg', help='cfg path')
    parser.add_argument('src', help='src lora model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    dst = Path(args.dst)
    if dst.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.cfg)
    lora_model = torch.load(args.src, map_location='cpu')

    merged_model = merge_lora_weight(cfg, lora_model)
    torch.save(merged_model, args.dst)

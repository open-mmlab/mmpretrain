# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers.modeling_utils import load_state_dict

prog_description = """\
Merge Llava delta weights and original weights,
and save as MMPreTrain checkpoint.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument(
        'src_path', type=str, help='The original checkpoint dir')
    parser.add_argument(
        'delta_path', type=str, help='The delta checkpoint dir')
    parser.add_argument('dst_path', type=str, help='The saved checkpoint path')
    args = parser.parse_args()
    return args


def load_checkpoint(path: Path):
    if path.is_file():
        return torch.load(path)

    state_dict = OrderedDict()
    for ckpt in chain(
            path.rglob('*.bin'), path.rglob('*.pth'),
            path.rglob('*.safetensors')):
        state_dict.update(load_state_dict(str(ckpt)))

    return state_dict


def main():
    args = parse_args()

    if Path(args.src_path).exists():
        src_path = Path(args.src_path)
    else:
        src_path = Path(snapshot_download(args.src_path))
    src_state_dict = load_checkpoint(src_path)

    if Path(args.delta_path).exists():
        delta_path = Path(args.delta_path)
    else:
        delta_path = Path(snapshot_download(args.delta_path))
    delta_state_dict = load_checkpoint(delta_path)

    merged_state_dict = OrderedDict()
    for k, v in src_state_dict.items():
        if k in delta_state_dict:
            delta_v = delta_state_dict.pop(k)
            if k in ['model.embed_tokens.weight', 'lm_head.weight']:
                h, w = v.shape[:2]
                delta_v[:h, :w] += v
                v = delta_v
            else:
                v += delta_v
        merged_state_dict['model.lang_encoder.' + k] = v

    for k, v in delta_state_dict.items():
        merged_state_dict['model.lang_encoder.' + k] = v

    torch.save(merged_state_dict, args.dst_path)
    print('Done!!')


if __name__ == '__main__':
    main()

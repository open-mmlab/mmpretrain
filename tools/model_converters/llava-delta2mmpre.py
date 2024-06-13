# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers.modeling_utils import load_state_dict

prog_description = """\
Convert Llava weights and original weights.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('src', type=str, help='The original checkpoint dir')
    parser.add_argument('dst', type=str, help='The saved checkpoint path')
    parser.add_argument('--delta', type=str, help='The delta checkpoint dir')
    args = parser.parse_args()
    return args


def load_checkpoint(path: Path):
    path = Path(path)
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

    if Path(args.src).exists():
        src_path = args.src
    else:
        src_path = snapshot_download(
            args.src, allow_patterns='pytorch_model*.bin')
    src_state_dict = load_checkpoint(src_path)

    if args.delta is None:
        delta_state_dict = {}
    elif Path(args.delta).exists():
        delta_state_dict = load_checkpoint(args.delta)
    else:
        delta_path = snapshot_download(
            args.delta, allow_patterns='pytorch_model*.bin')
        delta_state_dict = load_checkpoint(delta_path)

    new_state_dict = OrderedDict()
    for k, v in src_state_dict.items():
        if k in delta_state_dict:
            delta_v = delta_state_dict.pop(k)
            if k in ['model.embed_tokens.weight', 'lm_head.weight']:
                h, w = v.shape[:2]
                delta_v[:h, :w] += v
                v = delta_v
            else:
                v += delta_v
        if 'rotary_emb.inv_freq' not in k:
            new_state_dict['model.lang_encoder.' + k] = v

    for k, v in delta_state_dict.items():
        new_state_dict['model.lang_encoder.' + k] = v

    torch.save(new_state_dict, args.dst)
    print('Done!!')


if __name__ == '__main__':
    main()

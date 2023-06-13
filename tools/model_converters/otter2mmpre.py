# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import re
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import torch

prog_description = """\
Convert Official Otter HF models to MMPreTrain format.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument(
        'name_or_dir', type=str, help='The Otter HF model name or directory.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not Path(args.name_or_dir).is_dir():
        from huggingface_hub import snapshot_download
        ckpt_dir = Path(
            snapshot_download(args.name_or_dir, allow_patterns='*.bin'))
        name = args.name_or_dir.replace('/', '_')
    else:
        ckpt_dir = Path(args.name_or_dir)
        name = ckpt_dir.name

    state_dict = OrderedDict()
    for k, v in chain.from_iterable(
            torch.load(ckpt).items() for ckpt in ckpt_dir.glob('*.bin')):
        adapter_patterns = [
            r'^perceiver',
            r'lang_encoder.*embed_tokens',
            r'lang_encoder.*gated_cross_attn_layer',
            r'lang_encoder.*rotary_emb',
        ]
        if not any(re.match(pattern, k) for pattern in adapter_patterns):
            # Drop encoder parameters to decrease the size.
            continue

        # The keys are different between Open-Flamingo and Otter
        if 'gated_cross_attn_layer.feed_forward' in k:
            k = k.replace('feed_forward', 'ff')
        if 'perceiver.layers' in k:
            prefix_match = re.match(r'perceiver.layers.\d+.', k)
            prefix = k[:prefix_match.end()]
            suffix = k[prefix_match.end():]
            if 'feed_forward' in k:
                k = prefix + '1.' + suffix.replace('feed_forward.', '')
            else:
                k = prefix + '0.' + suffix
        state_dict[k] = v
    if len(state_dict) == 0:
        raise RuntimeError('No checkpoint found in the specified directory.')

    torch.save(state_dict, name + '.pth')


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import re
from collections import OrderedDict, namedtuple
from pathlib import Path

import torch

prog_description = """\
Convert OFA official models to MMPretrain format.
"""

MapItem = namedtuple(
    'MapItem', 'pattern repl key_action value_action', defaults=[None] * 4)


def convert_by_mapdict(src_dict: dict, map_dict: Path):
    dst_dict = OrderedDict()
    convert_map_dict = dict()

    for k, v in src_dict.items():
        ori_k = k
        for item in map_dict:
            pattern = item.pattern
            assert pattern is not None
            match = next(re.finditer(pattern, k), None)
            if match is None:
                continue
            match_group = match.groups()
            repl = item.repl

            key_action = item.key_action
            if key_action is not None:
                assert callable(key_action)
                match_group = key_action(*match_group)
                if isinstance(match_group, str):
                    match_group = (match_group, )
            start, end = match.span(0)
            if repl is not None:
                k = k[:start] + repl.format(*match_group) + k[end:]
            else:
                for i, sub in enumerate(match_group):
                    start, end = match.span(i + 1)
                    k = k[:start] + str(sub) + k[end:]

            value_action = item.value_action
            if value_action is not None:
                assert callable(value_action)
                v = value_action(v)

        if v is not None:
            dst_dict[k] = v
        convert_map_dict[k] = ori_k
    return dst_dict, convert_map_dict


map_dict = [
    # Encoder modules
    MapItem(r'\.type_embedding\.', '.embed_type.'),
    MapItem(r'\.layernorm_embedding\.', '.embedding_ln.'),
    MapItem(r'\.patch_layernorm_embedding\.', '.image_embedding_ln.'),
    MapItem(r'encoder.layer_norm\.', 'encoder.final_ln.'),
    # Encoder layers
    MapItem(r'\.attn_ln\.', '.attn_mid_ln.'),
    MapItem(r'\.ffn_layernorm\.', '.ffn_mid_ln.'),
    MapItem(r'\.final_layer_norm', '.ffn_ln'),
    MapItem(r'encoder.*(\.self_attn\.)', key_action=lambda _: '.attn.'),
    MapItem(
        r'encoder.*(\.self_attn_layer_norm\.)',
        key_action=lambda _: '.attn_ln.'),
    # Decoder modules
    MapItem(r'\.code_layernorm_embedding\.', '.code_embedding_ln.'),
    MapItem(r'decoder.layer_norm\.', 'decoder.final_ln.'),
    # Decoder layers
    MapItem(r'\.self_attn_ln', '.self_attn_mid_ln'),
    MapItem(r'\.cross_attn_ln', '.cross_attn_mid_ln'),
    MapItem(r'\.encoder_attn_layer_norm', '.cross_attn_ln'),
    MapItem(r'\.encoder_attn', '.cross_attn'),
    MapItem(
        r'decoder.*(\.self_attn_layer_norm\.)',
        key_action=lambda _: '.self_attn_ln.'),
    # Remove version key
    MapItem(r'version', '', value_action=lambda _: None),
    # Add model prefix
    MapItem(r'^', 'model.'),
]


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('src', type=str, help='The official checkpoint path.')
    parser.add_argument('dst', type=str, help='The save path.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    src = torch.load(args.src)
    if 'extra_state' in src and 'ema' in src['extra_state']:
        print('Use EMA weights.')
        src = src['extra_state']['ema']
    else:
        src = src['model']
    dst, _ = convert_by_mapdict(src, map_dict)
    torch.save(dst, args.dst)
    print('Done!!')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from ckpt_tree import StateDictTree, ckpt_to_state_dict
from rich.progress import track
from scipy import stats

prog_description = """\
Compare the initialization distribution between state dicts by Kolmogorov-Smirnov test.
"""  # noqa: E501


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=prog_description)
    parser.add_argument(
        'model_a',
        type=Path,
        help='The path of the first checkpoint or model config.')
    parser.add_argument(
        'model_b',
        type=Path,
        help='The path of the second checkpoint or model config.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to draw the KDE of variables')
    parser.add_argument(
        '-p',
        default=0.01,
        type=float,
        help='The threshold of p-value. '
        'Higher threshold means more strict test.')
    args = parser.parse_args()
    return args


def compare_distribution(state_dict_a, state_dict_b, p_thres):
    assert len(state_dict_a) == len(state_dict_b)
    for k, v1 in state_dict_a.items():
        assert k in state_dict_b
        v2 = state_dict_b[k]
        v1 = v1.cpu().flatten()
        v2 = v2.cpu().flatten()
        pvalue = stats.kstest(v1, v2).pvalue
        if pvalue < p_thres:
            yield k, pvalue, v1, v2


def state_dict_from_cfg_or_ckpt(path, state_key=None):
    if path.suffix in ['.json', '.py', '.yml']:
        from mmengine.runner import get_state_dict

        from mmpretrain.apis import init_model
        model = init_model(path, device='cpu')
        model.init_weights()
        return get_state_dict(model)
    else:
        ckpt = torch.load(path, map_location='cpu')
        return ckpt_to_state_dict(ckpt, state_key)


def main():
    args = parse_args()

    state_dict_a = state_dict_from_cfg_or_ckpt(args.model_a)
    state_dict_b = state_dict_from_cfg_or_ckpt(args.model_b)
    compare_keys = state_dict_a.keys() & state_dict_b.keys()
    if len(compare_keys) == 0:
        raise ValueError("The state dicts don't match, please convert "
                         'to the same keys before comparison.')

    root = StateDictTree()
    for key in track(compare_keys):
        if state_dict_a[key].shape != state_dict_b[key].shape:
            raise ValueError(f'The shapes of "{key}" are different. '
                             'Please check models in the same architecture.')

        # Sample at most 30000 items to prevent long-time calcuation.
        perm_ids = torch.randperm(state_dict_a[key].numel())[:30000]
        value_a = state_dict_a[key].flatten()[perm_ids]
        value_b = state_dict_b[key].flatten()[perm_ids]
        pvalue = stats.kstest(value_a, value_b).pvalue
        if pvalue < args.p:
            root.add_parameter(key, round(pvalue, 4))
            if args.show:
                try:
                    import seaborn as sns
                except ImportError:
                    raise ImportError('Please install `seaborn` by '
                                      '`pip install seaborn` to show KDE.')
                sample_a = str([round(v.item(), 2) for v in value_a[:10]])
                sample_b = str([round(v.item(), 2) for v in value_b[:10]])
                if value_a.std() > 0:
                    sns.kdeplot(value_a, fill=True)
                else:
                    sns.scatterplot(x=[value_a[0].item()], y=[1])
                if value_b.std() > 0:
                    sns.kdeplot(value_b, fill=True)
                else:
                    sns.scatterplot(x=[value_b[0].item()], y=[1])
                plt.legend([
                    f'{args.model_a.stem}: {sample_a}',
                    f'{args.model_b.stem}: {sample_b}'
                ])
                plt.title(key)
                plt.show()
    if len(root) > 0:
        root.draw_tree(with_value=True)
        print("Above parameters didn't pass the test, "
              'and the values are their similarity score.')
    else:
        print('The distributions of all weights are the same.')


if __name__ == '__main__':
    main()

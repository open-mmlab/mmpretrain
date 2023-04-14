import argparse
import math
from pathlib import Path

import torch
from rich.console import Console

console = Console()

prog_description = """\
Draw the state dict tree.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument(
        'path',
        type=Path,
        help='The path of the checkpoint or model config to draw.')
    parser.add_argument('--depth', type=int, help='The max depth to draw.')
    parser.add_argument(
        '--full-name',
        action='store_true',
        help='Whether to print the full name of the key.')
    parser.add_argument(
        '--shape',
        action='store_true',
        help='Whether to print the shape of the parameter.')
    parser.add_argument(
        '--state-key',
        type=str,
        help='The key of the state dict in the checkpoint.')
    parser.add_argument(
        '--number',
        action='store_true',
        help='Mark all parameters and their index number.')
    parser.add_argument(
        '--node',
        type=str,
        help='Show the sub-tree of a node, like "backbone.layers".')
    args = parser.parse_args()
    return args


def ckpt_to_state_dict(checkpoint, key=None):
    if key is not None:
        state_dict = checkpoint[key]
    elif 'state_dict' in checkpoint:
        # try mmpretrain style
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(next(iter(checkpoint.values())), torch.Tensor):
        # try native style
        state_dict = checkpoint
    else:
        raise KeyError('Please specify the key of state '
                       f'dict from {list(checkpoint.keys())}.')
    return state_dict


class StateDictTree:

    def __init__(self, key='', value=None):
        self.children = {}
        self.key: str = key
        self.value = value

    def add_parameter(self, key, value):
        keys = key.split('.', 1)
        if len(keys) == 1:
            self.children[key] = StateDictTree(key, value)
        elif keys[0] in self.children:
            self.children[keys[0]].add_parameter(keys[1], value)
        else:
            node = StateDictTree(keys[0])
            node.add_parameter(keys[1], value)
            self.children[keys[0]] = node

    def __getitem__(self, key: str):
        return self.children[key]

    def __repr__(self) -> str:
        with console.capture() as capture:
            for line in self.iter_tree():
                console.print(line)
        return capture.get()

    def __len__(self):
        return len(self.children)

    def draw_tree(self,
                  max_depth=None,
                  full_name=False,
                  with_shape=False,
                  with_value=False):
        for line in self.iter_tree(
                max_depth=max_depth,
                full_name=full_name,
                with_shape=with_shape,
                with_value=with_value,
        ):
            console.print(line, highlight=False)

    def iter_tree(
        self,
        lead='',
        prefix='',
        max_depth=None,
        full_name=False,
        with_shape=False,
        with_value=False,
    ):
        if self.value is None:
            key_str = f'[blue]{self.key}[/]'
        elif with_shape:
            key_str = f'[green]{self.key}[/] {tuple(self.value.shape)}'
        elif with_value:
            key_str = f'[green]{self.key}[/] {self.value}'
        else:
            key_str = f'[green]{self.key}[/]'

        yield lead + prefix + key_str

        lead = lead.replace('├─', '│ ')
        lead = lead.replace('└─', '  ')
        if self.key and full_name:
            prefix = f'{prefix}{self.key}.'

        if max_depth == 0:
            return
        elif max_depth is not None:
            max_depth -= 1

        for i, child in enumerate(self.children.values()):
            level_lead = '├─' if i < len(self.children) - 1 else '└─'
            yield from child.iter_tree(
                lead=f'{lead}{level_lead} ',
                prefix=prefix,
                max_depth=max_depth,
                full_name=full_name,
                with_shape=with_shape,
                with_value=with_value)


def main():
    args = parse_args()
    if args.path.suffix in ['.json', '.py', '.yml']:
        from mmengine.runner import get_state_dict

        from mmpretrain.apis import init_model
        model = init_model(args.path, device='cpu')
        state_dict = get_state_dict(model)
    else:
        ckpt = torch.load(args.path, map_location='cpu')
        state_dict = ckpt_to_state_dict(ckpt, args.state_key)

    root = StateDictTree()
    for k, v in state_dict.items():
        root.add_parameter(k, v)

    para_index = 0
    mark_width = math.floor(math.log(len(state_dict), 10) + 1)
    if args.node is not None:
        for key in args.node.split('.'):
            root = root[key]

    for line in root.iter_tree(
            max_depth=args.depth,
            full_name=args.full_name,
            with_shape=args.shape,
    ):
        if not args.number:
            mark = ''
        # A hack method to determine whether a line is parameter.
        elif '[green]' in line:
            mark = f'[red]({str(para_index).ljust(mark_width)})[/]'
            para_index += 1
        else:
            mark = ' ' * (mark_width + 2)
        console.print(mark + line, highlight=False)


if __name__ == '__main__':
    main()

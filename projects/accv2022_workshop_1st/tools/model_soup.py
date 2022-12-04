# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Model Soup')
    parser.add_argument(
        '--models', nargs='+', default=[], help='Ensemble results')
    parser.add_argument(
        '--model-folder', default=None, help='Ensemble results')
    parser.add_argument(
        '--out', default='uniform_soup.pth', help='output path')
    args = parser.parse_args()
    assert len(args.models) != 0 or args.model_folder is not None
    return args


def get_models(args):
    if args.models and args.model_folder:
        raise ValueError(
            'You can only use one of ``--models`` or `--model-folder`')
    if len(args.models) != 0:
        for m in args.models:
            assert m.endswith('.pth')
        return args.models
    else:
        files = os.listdir(args.model_folder)
        return [
            os.path.join(args.model_folder, f) for f in files
            if f.endswith('.pth')
        ]


def main():
    args = parse_args()
    model_paths = get_models(args)
    NUM_MODELS = len(model_paths)
    print(f'Find {NUM_MODELS} model to do model soup...')

    # create the uniform soup sequentially to not overload memory
    for j, model_path in enumerate(model_paths):

        print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

        assert os.path.exists(model_path), f'Can not find {model_path}'
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # for k, v in state_dict.items():
        #    print(k, type(v))
        if j == 0:
            uniform_soup = {
                k: v * (1. / NUM_MODELS)
                for k, v in state_dict.items()
            }
        else:
            uniform_soup = {
                k: v * (1. / NUM_MODELS) + uniform_soup[k]
                for k, v in state_dict.items()
            }

    torch.save(uniform_soup, args.out)


if __name__ == '__main__':
    main()

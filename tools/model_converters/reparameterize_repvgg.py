# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
from pathlib import Path

import torch

from mmcls.apis import init_model

bright_style, reset_style = '\x1b[1m', '\x1b[0m'
red_text, blue_text = '\x1b[31m', '\x1b[34m'
white_background = '\x1b[107m'

msg = bright_style + red_text
msg += 'DeprecationWarning: This tool will be deprecated in future. '
msg += red_text + 'Welcome to use the '
msg += white_background
msg += '"tools/convert_models/reparameterize_model.py"'
msg += reset_style
warnings.warn(msg)


def convert_repvggblock_param(config_path, checkpoint_path, save_path):
    model = init_model(config_path, checkpoint=checkpoint_path)
    print('Converting...')

    model.backbone.switch_to_deploy()
    torch.save(model.state_dict(), save_path)

    print('Done! Save at path "{}"'.format(save_path))


def main():
    parser = argparse.ArgumentParser(
        description='Convert the parameters of the repvgg block '
        'from training mode to deployment mode.')
    parser.add_argument(
        'config_path',
        help='The path to the configuration file of the network '
        'containing the repvgg block.')
    parser.add_argument(
        'checkpoint_path',
        help='The path to the checkpoint file corresponding to the model.')
    parser.add_argument(
        'save_path',
        help='The path where the converted checkpoint file is stored.')
    args = parser.parse_args()

    save_path = Path(args.save_path)
    if save_path.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit(1)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    convert_repvggblock_param(args.config_path, args.checkpoint_path,
                              args.save_path)


if __name__ == '__main__':
    main()

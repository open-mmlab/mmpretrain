import argparse
from pathlib import Path

import torch
from torch.nn.modules.batchnorm import BatchNorm2d

from mmcls.apis import init_model


def convert_repvggblock_param(config_path,
                              checkpoint_path,
                              save_path,
                              device='cuda'):
    model = init_model(config_path, checkpoint=checkpoint_path, device=device)

    print('Converting...')

    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            assert isinstance(module.branch_3x3.norm, BatchNorm2d), \
                'The normalization method in RepVGGBlock needs to ' \
                'be BatchNorm2d.'
            module.switch_to_deploy()

    torch.save(model.state_dict(), save_path)

    print('Done!')


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
    parser.add_argument(
        '--device',
        default='cuda',
        help='The device to which the model is loaded.')
    args = parser.parse_args()

    save_path = Path(args.save_path)
    if save_path.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    convert_repvggblock_param(
        args.config_path,
        args.checkpoint_path,
        args.save_path,
        device=args.device)


if __name__ == '__main__':
    main()

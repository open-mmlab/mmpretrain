# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from torch import nn

from mmcls.models import build_classifier

torch.manual_seed(3)


def _demo_mm_inputs(input_shape: tuple, num_classes: int):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(False),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def pytorch2torchscript(model: nn.Module, input_shape: tuple, output_file: str,
                        verify: bool):
    """Export Pytorch model to TorchScript model through torch.jit.trace and
    verify the outputs are same between Pytorch and TorchScript.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output
            TorchScript model.
        verify (bool): Whether compare the outputs between Pytorch
            and TorchScript through loading generated output_file.
    """
    model.cpu().eval()

    num_classes = model.head.num_classes
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :] for img in imgs]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas={}, return_loss=False)

    with torch.no_grad():
        trace_model = torch.jit.trace(model, img_list[0])
        save_dir, _ = osp.split(output_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        trace_model.save(output_file)
        print(f'Successfully exported TorchScript model: {output_file}')
    model.forward = origin_forward

    if verify:
        # load by torch.jit
        jit_model = torch.jit.load(output_file)

        # check the numerical value
        # get pytorch output
        pytorch_result = model(img_list, img_metas={}, return_loss=False)[0]

        # get jit output
        jit_result = jit_model(img_list[0])[0].detach().numpy()
        if not np.allclose(pytorch_result, jit_result):
            raise ValueError(
                'The outputs are different between Pytorch and TorchScript')
        print('The outputs are same between Pytorch and TorchScript')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMCls to TorchScript')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the TorchScript model',
        default=False)
    parser.add_argument('--output-file', type=str, default='tmp.pt')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)

    if args.checkpoint:
        load_checkpoint(classifier, args.checkpoint, map_location='cpu')

    # convert model to TorchScript file
    pytorch2torchscript(
        classifier,
        input_shape,
        output_file=args.output_file,
        verify=args.verify)

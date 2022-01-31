# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from torch import nn

from mmcls.models import build_classifier

torch.manual_seed(3)

try:
    import coremltools as ct
except ImportError:
    raise ImportError('Please install coremltools to enable output file.')


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


def pytorch2mlmodel(model: nn.Module, input_shape: tuple, output_file: str,
                    add_norm: bool, norm: dict):
    """Export Pytorch model to mlmodel format that can be deployed in apple
    devices through torch.jit.trace and the coremltools library.

       Optionally, embed the normalization step as a layer to the model.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output
            TorchScript model.
        add_norm (bool): Whether to embed the normalization layer to the
            output model.
        norm (dict): image normalization config for embedding it as a layer
            to the output model.
    """
    model.cpu().eval()

    num_classes = model.head.num_classes
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :] for img in imgs]
    model.forward = partial(model.forward, img_metas={}, return_loss=False)

    with torch.no_grad():
        trace_model = torch.jit.trace(model, img_list[0])
        save_dir, _ = osp.split(output_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if add_norm:
            means, stds = norm.mean, norm.std
            if stds.count(stds[0]) != len(stds):
                warnings.warn(f'Image std from config is {stds}. However, '
                              'current version of coremltools (5.1) uses a '
                              'global std rather than the channel-specific '
                              'values that torchvision uses. A mean will be '
                              'taken but this might tamper with the resulting '
                              'model\'s predictions. For more details refer '
                              'to the coreml docs on ImageType pre-processing')
                scale = np.mean(stds)
            else:
                scale = stds[0]

            bias = [-mean / scale for mean in means]
            image_input = ct.ImageType(
                name='input_1',
                shape=input_shape,
                scale=1 / scale,
                bias=bias,
                color_layout='RGB',
                channel_first=True)

            coreml_model = ct.convert(trace_model, inputs=[image_input])
            coreml_model.save(output_file)
        else:
            coreml_model = ct.convert(
                trace_model, inputs=[ct.TensorType(shape=input_shape)])
            coreml_model.save(output_file)

        print(f'Successfully exported coreml model: {output_file}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMCls to MlModel format for apple devices')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--output-file', type=str, default='model.mlmodel')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument(
        '--add-norm-layer',
        action='store_true',
        help='embed normalization layer to deployed model')
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

    # convert model to mlmodel file
    pytorch2mlmodel(
        classifier,
        input_shape,
        output_file=args.output_file,
        add_norm=args.add_norm_layer,
        norm=cfg.img_norm_cfg)

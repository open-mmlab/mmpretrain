# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import numpy as np
import torch
from mmcv.runner import Sequential
from tensorflow.python.training import py_checkpoint_reader

from mmcls.models.backbones.efficientnet import EfficientNet


def tf2pth(v):
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3, 2, 0, 1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v


def read_ckpt(ckpt):
    reader = py_checkpoint_reader.NewCheckpointReader(ckpt)
    weights = {
        n: torch.as_tensor(tf2pth(reader.get_tensor(n)))
        for (n, _) in reader.get_variable_to_shape_map().items()
    }
    return weights


def map_key(weight):
    m = dict()
    has_expand_conv = set()
    is_MBConv = set()
    max_idx = 0
    name = None
    for k, v in weight.items():
        seg = k.split('/')
        if len(seg) == 1:
            continue
        if 'edgetpu' in seg[0]:
            name = 'e' + seg[0][21:].lower()
        else:
            name = seg[0][13:]
        if seg[2] == 'tpu_batch_normalization_2':
            has_expand_conv.add(seg[1])
        if seg[1].startswith('blocks_'):
            idx = int(seg[1][7:]) + 1
            max_idx = max(max_idx, idx)
            if 'depthwise' in k:
                is_MBConv.add(seg[1])

    model = EfficientNet(name)
    idx2key = []
    for idx, module in enumerate(model.layers):
        if isinstance(module, Sequential):
            for j in range(len(module)):
                idx2key.append('{}.{}'.format(idx, j))
        else:
            idx2key.append('{}'.format(idx))

    for k, v in weight.items():

        if 'Exponential' in k or 'RMS' in k:
            continue

        seg = k.split('/')
        if len(seg) == 1:
            continue
        if seg[2] == 'depthwise_conv2d':
            v = v.transpose(1, 0)

        if seg[1] == 'stem':
            prefix = 'backbone.layers.{}'.format(idx2key[0])
            mapping = {
                'conv2d/kernel': 'conv.weight',
                'tpu_batch_normalization/beta': 'bn.bias',
                'tpu_batch_normalization/gamma': 'bn.weight',
                'tpu_batch_normalization/moving_mean': 'bn.running_mean',
                'tpu_batch_normalization/moving_variance': 'bn.running_var',
            }
            suffix = mapping['/'.join(seg[2:])]
            m[prefix + '.' + suffix] = v

        elif seg[1].startswith('blocks_'):
            idx = int(seg[1][7:]) + 1
            prefix = '.'.join(['backbone', 'layers', idx2key[idx]])
            if seg[1] not in is_MBConv:
                mapping = {
                    'conv2d/kernel':
                    'conv1.conv.weight',
                    'tpu_batch_normalization/gamma':
                    'conv1.bn.weight',
                    'tpu_batch_normalization/beta':
                    'conv1.bn.bias',
                    'tpu_batch_normalization/moving_mean':
                    'conv1.bn.running_mean',
                    'tpu_batch_normalization/moving_variance':
                    'conv1.bn.running_var',
                    'conv2d_1/kernel':
                    'conv2.conv.weight',
                    'tpu_batch_normalization_1/gamma':
                    'conv2.bn.weight',
                    'tpu_batch_normalization_1/beta':
                    'conv2.bn.bias',
                    'tpu_batch_normalization_1/moving_mean':
                    'conv2.bn.running_mean',
                    'tpu_batch_normalization_1/moving_variance':
                    'conv2.bn.running_var',
                }
            else:

                base_mapping = {
                    'depthwise_conv2d/depthwise_kernel':
                    'depthwise_conv.conv.weight',
                    'se/conv2d/kernel': 'se.conv1.conv.weight',
                    'se/conv2d/bias': 'se.conv1.conv.bias',
                    'se/conv2d_1/kernel': 'se.conv2.conv.weight',
                    'se/conv2d_1/bias': 'se.conv2.conv.bias'
                }

                if seg[1] not in has_expand_conv:
                    mapping = {
                        'conv2d/kernel':
                        'linear_conv.conv.weight',
                        'tpu_batch_normalization/beta':
                        'depthwise_conv.bn.bias',
                        'tpu_batch_normalization/gamma':
                        'depthwise_conv.bn.weight',
                        'tpu_batch_normalization/moving_mean':
                        'depthwise_conv.bn.running_mean',
                        'tpu_batch_normalization/moving_variance':
                        'depthwise_conv.bn.running_var',
                        'tpu_batch_normalization_1/beta':
                        'linear_conv.bn.bias',
                        'tpu_batch_normalization_1/gamma':
                        'linear_conv.bn.weight',
                        'tpu_batch_normalization_1/moving_mean':
                        'linear_conv.bn.running_mean',
                        'tpu_batch_normalization_1/moving_variance':
                        'linear_conv.bn.running_var',
                    }
                else:
                    mapping = {
                        'depthwise_conv2d/depthwise_kernel':
                        'depthwise_conv.conv.weight',
                        'conv2d/kernel':
                        'expand_conv.conv.weight',
                        'conv2d_1/kernel':
                        'linear_conv.conv.weight',
                        'tpu_batch_normalization/beta':
                        'expand_conv.bn.bias',
                        'tpu_batch_normalization/gamma':
                        'expand_conv.bn.weight',
                        'tpu_batch_normalization/moving_mean':
                        'expand_conv.bn.running_mean',
                        'tpu_batch_normalization/moving_variance':
                        'expand_conv.bn.running_var',
                        'tpu_batch_normalization_1/beta':
                        'depthwise_conv.bn.bias',
                        'tpu_batch_normalization_1/gamma':
                        'depthwise_conv.bn.weight',
                        'tpu_batch_normalization_1/moving_mean':
                        'depthwise_conv.bn.running_mean',
                        'tpu_batch_normalization_1/moving_variance':
                        'depthwise_conv.bn.running_var',
                        'tpu_batch_normalization_2/beta':
                        'linear_conv.bn.bias',
                        'tpu_batch_normalization_2/gamma':
                        'linear_conv.bn.weight',
                        'tpu_batch_normalization_2/moving_mean':
                        'linear_conv.bn.running_mean',
                        'tpu_batch_normalization_2/moving_variance':
                        'linear_conv.bn.running_var',
                    }
                mapping.update(base_mapping)
            suffix = mapping['/'.join(seg[2:])]
            m[prefix + '.' + suffix] = v
        elif seg[1] == 'head':
            seq_key = idx2key[max_idx + 1]
            mapping = {
                'conv2d/kernel':
                'backbone.layers.{}.conv.weight'.format(seq_key),
                'tpu_batch_normalization/beta':
                'backbone.layers.{}.bn.bias'.format(seq_key),
                'tpu_batch_normalization/gamma':
                'backbone.layers.{}.bn.weight'.format(seq_key),
                'tpu_batch_normalization/moving_mean':
                'backbone.layers.{}.bn.running_mean'.format(seq_key),
                'tpu_batch_normalization/moving_variance':
                'backbone.layers.{}.bn.running_var'.format(seq_key),
                'dense/kernel':
                'head.fc.weight',
                'dense/bias':
                'head.fc.bias'
            }
            key = mapping['/'.join(seg[2:])]
            if name.startswith('e') and 'fc' in key:
                v = v[1:]
            m[key] = v
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='Path to the ckpt.')
    parser.add_argument('outfile', type=str, help='Output file.')
    args = parser.parse_args()
    assert args.outfile

    outdir = os.path.dirname(os.path.abspath(args.outfile))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    weights = read_ckpt(args.infile)
    weights = map_key(weights)
    torch.save(weights, args.outfile)

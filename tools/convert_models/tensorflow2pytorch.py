import argparse
import os

import numpy as np
import torch
from tensorflow.python.training import py_checkpoint_reader


def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3, 2, 0, 1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v


def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader = py_checkpoint_reader.NewCheckpointReader(ckpt)
    weights = {
        n: reader.get_tensor(n)
        for (n, _) in reader.get_variable_to_shape_map().items()
    }
    pyweights = {k: torch.as_tensor(tr(v)) for (k, v) in weights.items()}
    return pyweights


def map_key(weight):
    m = dict()
    has_expand_conv = set()
    max_idx = 0
    for k, v in weight.items():
        seg = k.split('/')
        if seg[2] == 'tpu_batch_normalization_2':
            has_expand_conv.add(seg[1])
        if seg[1].startswith('blocks_'):
            idx = int(seg[1][7:]) + 1
            max_idx = max(max_idx, idx)
    for k, v in weight.items():
        if 'Exponential' in k:
            continue
        seg = k.split('/')
        if seg[2] == 'depthwise_conv2d':
            v = v.transpose(1, 0)
        if seg[1] == 'stem':
            prefix = 'backbone.layers.0'
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
            prefix = '.'.join(['backbone', 'layers', str(idx)])

            if seg[1] not in has_expand_conv:
                mapping = {
                    'depthwise_conv2d/depthwise_kernel':
                    'depthwise_conv.conv.weight',
                    'conv2d/kernel': 'linear_conv.conv.weight',
                    'tpu_batch_normalization/beta': 'depthwise_conv.bn.bias',
                    'tpu_batch_normalization/gamma':
                    'depthwise_conv.bn.weight',
                    'tpu_batch_normalization/moving_mean':
                    'depthwise_conv.bn.running_mean',
                    'tpu_batch_normalization/moving_variance':
                    'depthwise_conv.bn.running_var',
                    'tpu_batch_normalization_1/beta': 'linear_conv.bn.bias',
                    'tpu_batch_normalization_1/gamma': 'linear_conv.bn.weight',
                    'tpu_batch_normalization_1/moving_mean':
                    'linear_conv.bn.running_mean',
                    'tpu_batch_normalization_1/moving_variance':
                    'linear_conv.bn.running_var',
                    'se/conv2d/kernel': 'se.conv1.conv.weight',
                    'se/conv2d/bias': 'se.conv1.conv.bias',
                    'se/conv2d_1/kernel': 'se.conv2.conv.weight',
                    'se/conv2d_1/bias': 'se.conv2.conv.bias'
                }
            else:
                mapping = {
                    'depthwise_conv2d/depthwise_kernel':
                    'depthwise_conv.conv.weight',
                    'conv2d/kernel': 'expand_conv.conv.weight',
                    'conv2d_1/kernel': 'linear_conv.conv.weight',
                    'tpu_batch_normalization/beta': 'expand_conv.bn.bias',
                    'tpu_batch_normalization/gamma': 'expand_conv.bn.weight',
                    'tpu_batch_normalization/moving_mean':
                    'expand_conv.bn.running_mean',
                    'tpu_batch_normalization/moving_variance':
                    'expand_conv.bn.running_var',
                    'tpu_batch_normalization_1/beta': 'depthwise_conv.bn.bias',
                    'tpu_batch_normalization_1/gamma':
                    'depthwise_conv.bn.weight',
                    'tpu_batch_normalization_1/moving_mean':
                    'depthwise_conv.bn.running_mean',
                    'tpu_batch_normalization_1/moving_variance':
                    'depthwise_conv.bn.running_var',
                    'tpu_batch_normalization_2/beta': 'linear_conv.bn.bias',
                    'tpu_batch_normalization_2/gamma': 'linear_conv.bn.weight',
                    'tpu_batch_normalization_2/moving_mean':
                    'linear_conv.bn.running_mean',
                    'tpu_batch_normalization_2/moving_variance':
                    'linear_conv.bn.running_var',
                    'se/conv2d/kernel': 'se.conv1.conv.weight',
                    'se/conv2d/bias': 'se.conv1.conv.bias',
                    'se/conv2d_1/kernel': 'se.conv2.conv.weight',
                    'se/conv2d_1/bias': 'se.conv2.conv.bias'
                }
            suffix = mapping['/'.join(seg[2:])]
            m[prefix + '.' + suffix] = v
        elif seg[1] == 'head':
            mapping = {
                'conv2d/kernel':
                'backbone.layers.{}.conv.weight'.format(max_idx + 1),
                'tpu_batch_normalization/beta':
                'backbone.layers.{}.bn.bias'.format(max_idx + 1),
                'tpu_batch_normalization/gamma':
                'backbone.layers.{}.bn.weight'.format(max_idx + 1),
                'tpu_batch_normalization/moving_mean':
                'backbone.layers.{}.bn.running_mean'.format(max_idx + 1),
                'tpu_batch_normalization/moving_variance':
                'backbone.layers.{}.bn.running_var'.format(max_idx + 1),
                'dense/kernel':
                'head.fc.weight',
                'dense/bias':
                'head.fc.bias'
            }
            key = mapping['/'.join(seg[2:])]
            m[key] = v
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='Path to the ckpt.')
    parser.add_argument(
        'outfile',
        type=str,
        nargs='?',
        default='',
        help='Output file (inferred if missing).')
    args = parser.parse_args()
    assert args.outfile

    outdir = os.path.dirname(os.path.abspath(args.outfile))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    weights = read_ckpt(args.infile)
    weights = map_key(weights)
    torch.save(weights, args.outfile)

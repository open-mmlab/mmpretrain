# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

import mmcv
import numpy as np
from mmcv import DictAction
from mmcv.parallel import MMDataParallel

from mmcls.apis import single_gpu_test
from mmcls.core.export import ONNXRuntimeClassifier, TensorRTClassifier
from mmcls.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test (and eval) an ONNX model using ONNXRuntime.')
    parser.add_argument('config', help='model config file')
    parser.add_argument('model', help='filename of the input ONNX model')
    parser.add_argument(
        '--backend',
        help='Backend of the model.',
        choices=['onnxruntime', 'tensorrt'])
    parser.add_argument(
        '--out', type=str, help='output result file in pickle format')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=False,
        round_up=False)

    # build onnxruntime model and run inference.
    if args.backend == 'onnxruntime':
        model = ONNXRuntimeClassifier(
            args.model, class_names=dataset.CLASSES, device_id=0)
    elif args.backend == 'tensorrt':
        model = TensorRTClassifier(
            args.model, class_names=dataset.CLASSES, device_id=0)
    else:
        print('Unknown backend: {}.'.format(args.model))
        exit(1)

    model = MMDataParallel(model, device_ids=[0])
    model.CLASSES = dataset.CLASSES
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)

    if args.metrics:
        results = dataset.evaluate(outputs, args.metrics, args.metric_options)
        for k, v in results.items():
            print(f'\n{k} : {v:.2f}')
    else:
        warnings.warn('Evaluation metrics are not specified.')
        scores = np.vstack(outputs)
        pred_score = np.max(scores, axis=1)
        pred_label = np.argmax(scores, axis=1)
        pred_class = [dataset.CLASSES[lb] for lb in pred_label]
        results = {
            'pred_score': pred_score,
            'pred_label': pred_label,
            'pred_class': pred_class
        }
        if not args.out:
            print('\nthe predicted result for the first element is '
                  f'pred_score = {pred_score[0]:.2f}, '
                  f'pred_label = {pred_label[0]} '
                  f'and pred_class = {pred_class[0]}. '
                  'Specify --out to save all results to files.')
    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

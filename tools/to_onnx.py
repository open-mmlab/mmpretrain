import argparse

import numpy as np
import onnxruntime as ort
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmcls.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--in_file', help='input checkpoint filename')
    parser.add_argument('--out_file', default='./model.onnx', help='output checkpoint filename')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_classifier(cfg.model)
    model.eval()

    if args.in_file is not None:
        print('load checkpoint...')
        load_checkpoint(model, args.in_file, map_location='cpu')

    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
                format(model.__class__.__name__))

    data_np = np.random.randn(*input_shape).astype(np.float32)
    data = torch.tensor(data_np)
    torch.onnx.export(model, data, args.out_file,
                      input_names=['input'], output_names=['output'])
    ort_session = ort.InferenceSession(args.out_file)
    result = model(data)
    print(result, result.shape)
    outputs = ort_session.run(None, {'input': data_np})
    print(outputs[0], outputs[0].shape)


if __name__ == '__main__':
    main()

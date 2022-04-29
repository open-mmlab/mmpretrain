# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings

import numpy as np


def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)


def onnx2tensorrt(onnx_file,
                  trt_file,
                  input_shape,
                  max_batch_size,
                  fp16_mode=False,
                  verify=False,
                  workspace_size=1):
    """Create tensorrt engine from onnx model.

    Args:
        onnx_file (str): Filename of the input ONNX model file.
        trt_file (str): Filename of the output TensorRT engine file.
        input_shape (list[int]): Input shape of the model.
            eg [1, 3, 224, 224].
        max_batch_size (int): Max batch size of the model.
        verify (bool, optional): Whether to verify the converted model.
            Defaults to False.
        workspace_size (int, optional): Maximum workspace of GPU.
            Defaults to 1.
    """
    import onnx
    from mmcv.tensorrt import TRTWraper, onnx2trt, save_trt_engine

    onnx_model = onnx.load(onnx_file)
    # create trt engine and wrapper
    assert max_batch_size >= 1
    max_shape = [max_batch_size] + list(input_shape[1:])
    opt_shape_dict = {'input': [input_shape, input_shape, max_shape]}
    max_workspace_size = get_GiB(workspace_size)
    trt_engine = onnx2trt(
        onnx_model,
        opt_shape_dict,
        fp16_mode=fp16_mode,
        max_workspace_size=max_workspace_size)
    save_dir, _ = osp.split(trt_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_trt_engine(trt_engine, trt_file)
    print(f'Successfully created TensorRT engine: {trt_file}')

    if verify:
        import onnxruntime as ort
        import torch

        input_img = torch.randn(*input_shape)
        input_img_cpu = input_img.detach().cpu().numpy()
        input_img_cuda = input_img.cuda()

        # Get results from ONNXRuntime
        session_options = ort.SessionOptions()
        sess = ort.InferenceSession(onnx_file, session_options)

        # get input and output names
        input_names = [_.name for _ in sess.get_inputs()]
        output_names = [_.name for _ in sess.get_outputs()]

        onnx_outputs = sess.run(None, {
            input_names[0]: input_img_cpu,
        })

        # Get results from TensorRT
        trt_model = TRTWraper(trt_file, input_names, output_names)
        with torch.no_grad():
            trt_outputs = trt_model({input_names[0]: input_img_cuda})
        trt_outputs = [
            trt_outputs[_].detach().cpu().numpy() for _ in output_names
        ]

        # Compare results
        np.testing.assert_allclose(
            onnx_outputs[0], trt_outputs[0], rtol=1e-05, atol=1e-05)
        print('The numerical values are the same ' +
              'between ONNXRuntime and TensorRT')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMClassification models from ONNX to TensorRT')
    parser.add_argument('model', help='Filename of the input ONNX model')
    parser.add_argument(
        '--trt-file',
        type=str,
        default='tmp.trt',
        help='Filename of the output TensorRT engine')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the outputs of ONNXRuntime and TensorRT')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='Input size of the model')
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=1,
        help='Maximum batch size of TensorRT model.')
    parser.add_argument('--fp16', action='store_true', help='Enable fp16 mode')
    parser.add_argument(
        '--workspace-size',
        type=int,
        default=1,
        help='Max workspace size of GPU in GiB')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    # Create TensorRT engine
    onnx2tensorrt(
        args.model,
        args.trt_file,
        input_shape,
        args.max_batch_size,
        fp16_mode=args.fp16,
        verify=args.verify,
        workspace_size=args.workspace_size)

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

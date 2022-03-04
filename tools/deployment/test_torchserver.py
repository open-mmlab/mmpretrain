# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import numpy as np
import requests

from mmcls.apis import inference_model, init_model, show_result_pyplot


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main(args):
    # Inference single image by native apis.
    model = init_model(args.config, args.checkpoint, device=args.device)
    model_result = inference_model(model, args.img)
    show_result_pyplot(model, args.img, model_result, title='pytorch_result')

    # Inference single image by torchserve engine.
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.img, 'rb') as image:
        response = requests.post(url, image)
    server_result = response.json()
    show_result_pyplot(model, args.img, server_result, title='server_result')

    assert np.allclose(model_result['pred_score'], server_result['pred_score'])
    print('Test complete, the results of PyTorch and TorchServe are the same.')


if __name__ == '__main__':
    args = parse_args()
    main(args)

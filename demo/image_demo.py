# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import torch

from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    x = torch.randn((1, 3, 224, 224)).to(args.device)
    y = model(x, return_loss=False)
    print(y[0].shape)
    # test a single image
    # result = inference_model(model, args.img)
    # show the results
    # show_result_pyplot(model, args.img, result)


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv import Config, DictAction

from mmcls.apis import init_model
from mmcls.datasets.pipelines import Compose

try:
    from pytorch_grad_cam import (AblationCAM, EigenCAM, GradCAM,
                                  GradCAMPlusPlus, ScoreCAM, XGradCAM)
    import pytorch_grad_cam.activations_and_gradients as act_and_grad
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print('please use `pip install grad-cam` to install pytorch_grad_cam')

# set of transforms, which just change data format, but not change the pictures
FORMAT_TRANSFORMS_SET = {'ToTensor', 'Normalize', 'ImageToTensor', 'Collect'}

# Supported grad-cam type map
GradCAM_MAP = {
    'GradCAM': GradCAM,
    'ScoreCAM': ScoreCAM,
    'GradCAM++': GradCAMPlusPlus,
    'AblationCAM': AblationCAM,
    'XGradCAM': XGradCAM,
    'EigenCAM': EigenCAM
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Cam Grad')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--preview',
        default=False,
        action='store_true',
        help='preview the model')
    parser.add_argument(
        '--cam-type',
        default='GradCAM',
        choices=list(GradCAM_MAP.keys()),
        help='Type of algorithm to use, support "GradCAM", "ScoreCAM",'
        '"GradCAM++", "AblationCAM", "XGradCAM" and "EigenCAM".')
    parser.add_argument(
        '--target-layers',
        default=[],
        nargs='+',
        type=str,
        help='target layers in insight')
    parser.add_argument(
        '--target-category',
        default=None,
        type=int,
        help='target category in insight')
    parser.add_argument(
        '--save-path',
        type=Path,
        help='The learning rate curve plot save path')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    return args


def apply_transforms(img_path, pipeline_cfg):
    """Since there are some transforms, which will change the regin to
    inference such as CenterCrop.

    So it is necessary to get inference image.
    """
    data = dict(img_info=dict(filename=img_path), img_prefix=None)

    def split_pipeline_cfg(pipeline_cfg):
        """to split the transfoms into image_transforms and
        format_transforms."""
        image_transforms_cfg, format_transforms_cfg = [], []
        if pipeline_cfg[0]['type'] != 'LoadImageFromFile':
            pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
        for transform in pipeline_cfg:
            if transform['type'] in FORMAT_TRANSFORMS_SET:
                format_transforms_cfg.append(transform)
            else:
                image_transforms_cfg.append(transform)
        return image_transforms_cfg, format_transforms_cfg

    image_transforms, format_transforms = split_pipeline_cfg(pipeline_cfg)
    image_transforms = Compose(image_transforms)
    format_transforms = Compose(format_transforms)

    intermediate_data = image_transforms(data)
    inference_img = copy.deepcopy(intermediate_data['img'])
    format_data = format_transforms(intermediate_data)

    return format_data, inference_img


def init_cam(cam_type, model, target_layers, use_cuda):
    """Construct the CAM object once, In order to be compatible with mmcls,
    here we modify the ActivationsAndGradients object and the get_loss
    function."""

    class mmActivationsAndGradients(act_and_grad.ActivationsAndGradients):
        """since the original __call__ can not pass additional parameters we
        modify the function to return torch.tensor."""

        def __call__(self, x):
            self.gradients = []
            self.activations = []

            return self.model(x, return_loss=False, return_score=True)

    def custom_get_loss(self, output, target_category):
        """get loss function of MMCls model."""
        loss = 0
        for i in range(len(target_category)):
            label = torch.tensor([target_category[i]])
            loss += self.model.head.loss(output, label)['loss']
        return loss

    GradCAM_Class = GradCAM_MAP[cam_type]
    MMGradCAM = type('MMGradCAM', (GradCAM_Class, ),
                     {'get_loss': custom_get_loss})
    cam = MMGradCAM(
        model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam.activations_and_grads = mmActivationsAndGradients(
        cam.model, cam.target_layers, cam.reshape_transform)

    return cam


def get_layer(layer_str, model):
    """get model lyaer from given str."""
    cur_layer = model
    assert layer_str.startswith('model'), "target-layer must start with 'model"
    layer_items = layer_str.strip().split('.')
    assert not layer_items[-1].startswith(
        'relu') and not layer_items[-1].startswith('bn')
    for item_str in layer_items[1:]:
        if hasattr(cur_layer, item_str):
            cur_layer = getattr(cur_layer, item_str)
        else:
            raise ValueError(
                f"model don't have `{layer_str}`, please use valid layers")
    return cur_layer


def show_cam_grad(grayscale_cam, src_img, title='cam-grad', out_path=None):
    """fuse src_img and grayscale_cam and show or save."""
    grayscale_cam = grayscale_cam[0, :]
    src_img = np.float32(src_img[:, :, ::-1]) / 255
    visualization_img = show_cam_on_image(src_img, grayscale_cam)
    if out_path:
        cv2.imsave(out_path, visualization_img)
    else:
        cv2.imshow(title, visualization_img)
        cv2.waitKey()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the model from a config file and a checkpoint file
    model = init_model(cfg, args.checkpoint, device=args.device)
    # if don't know the layers of a model, preview all layers in a model
    if args.preview:
        print(model)
        sys.exit()

    # apply transform and perpare data
    data, src_img = apply_transforms(args.img, cfg.data.test.pipeline)
    data['img'] = data['img'].unsqueeze(0)

    # build target layers
    target_layers = []
    for layer_str in args.target_layers:
        target_layers.append(get_layer(layer_str, model))
    assert len(args.target_layers) != 0, '`--target-layers` can not be empty'

    # init a cam grad computer
    use_cuda = True if 'cuda' in args.device else False
    cam = init_cam(args.cam_type, model, target_layers, use_cuda)

    # calculate cam grads and show|save
    grayscale_cam = cam(
        input_tensor=data['img'], target_category=args.target_category)
    show_cam_grad(
        grayscale_cam, src_img, title=args.cam_type, out_path=args.save_path)


if __name__ == '__main__':
    main()

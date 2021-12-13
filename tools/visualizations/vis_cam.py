# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import sys
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config, DictAction
from mmcv.utils import to_2tuple

from mmcls.apis import init_model
from mmcls.datasets.pipelines import Compose
from mmcls.models.backbones import SwinTransformer, T2T_ViT, VisionTransformer

try:
    from pytorch_grad_cam import (EigenCAM, GradCAM, GradCAMPlusPlus, XGradCAM,
                                  EigenGradCAM, LayerCAM)
    import pytorch_grad_cam.activations_and_gradients as act_and_grad
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError(
        'please use `pip install grad-cam` to install pytorch_grad_cam')

# set of transforms, which just change data format, not change the pictures
FORMAT_TRANSFORMS_SET = {'ToTensor', 'Normalize', 'ImageToTensor', 'Collect'}

# Supported grad-cam type map
METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}

# Transformer set based on ViT
ViT_based_Transformers = tuple([T2T_ViT, VisionTransformer])

# Transformer set based on Swin
Swin_based_Transformers = tuple([SwinTransformer])


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--target-layers',
        default=[],
        nargs='+',
        type=str,
        help='The target layers to get CAM')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--method',
        default='GradCAM',
        help='Type of method to use, supports '
        f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument(
        '--target-category',
        default=None,
        type=int,
        help='The target category to get CAM, default to use result '
        'get from given model.')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
        '``cam_weights*activations``')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument(
        '--save-path',
        type=Path,
        help='The path to save visualize cam image, default not to save.')
    parser.add_argument('--device', default='cpu', help='Device to use cpu')
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
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args


def build_reshape_transform(model):
    """build reshape_transform for `cam.activations_and_grads`, some neural
    networks such as SwinTransformer and VisionTransformer need an additional
    reshape operation.

    CNNs don't need, jush return `None`.
    """
    # ViT_based_Transformers have an additional clstoken in features
    if isinstance(model.backbone, Swin_based_Transformers):
        has_clstoken = False
    elif isinstance(model.backbone, ViT_based_Transformers):
        has_clstoken = True
    else:
        return None

    def _reshape_transform(tensor, has_clstoken=has_clstoken):
        """reshape_transform helper."""
        tensor = tensor[:, 1:, :] if has_clstoken else tensor
        # get heat_map_height and heat_map_width, preset input is a square
        heat_map_area = tensor.size()[1]
        height, width = to_2tuple(int(math.sqrt(heat_map_area)))
        message = 'Only square input images are supported for Transformers.'
        assert height * height == heat_map_area, message
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # Bring the channels to the first dimension, like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    return _reshape_transform


def apply_transforms(img_path, pipeline_cfg):
    """Since there are some transforms, which will change the regin to
    inference such as CenterCrop.

    So it is necessary to get inference image. this function is to get
    transformed imgaes besides transformes in `FORMAT_TRANSFORMS_SET`
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


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once, In order to be compatible with mmcls,
    here we modify the ActivationsAndGradients object."""

    class mmActivationsAndGradients(act_and_grad.ActivationsAndGradients):
        """since the original __call__ can not pass additional parameters we
        modify the function to return torch.tensor."""

        def __call__(self, x):
            self.gradients = []
            self.activations = []

            return self.model(
                x, return_loss=False, softmax=False, post_process=False)

    GradCAM_Class = METHOD_MAP[method.lower()]
    cam = GradCAM_Class(
        model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam.activations_and_grads = mmActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)

    return cam


def get_layer(layer_str, model):
    """get model lyaer from given str."""
    cur_layer = model
    assert layer_str.startswith(
        'model'), "target-layer must start with 'model'"
    layer_items = layer_str.strip().split('.')
    assert not (layer_items[-1].startswith('relu')
                or layer_items[-1].startswith('bn')
                ), "target-layer can't be 'bn' or 'relu'"
    for item_str in layer_items[1:]:
        if hasattr(cur_layer, item_str):
            cur_layer = getattr(cur_layer, item_str)
        else:
            raise ValueError(
                f"model don't have `{layer_str}`, please use valid layers")
    return cur_layer


def show_cam_grad(grayscale_cam, src_img, title, out_path=None):
    """fuse src_img and grayscale_cam and show or save."""
    grayscale_cam = grayscale_cam[0, :]
    src_img = np.float32(src_img) / 255
    visualization_img = show_cam_on_image(
        src_img, grayscale_cam, use_rgb=False)

    if out_path:
        mmcv.imwrite(visualization_img, str(out_path))
    else:
        mmcv.imshow(visualization_img, win_name=title)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the model from a config file and a checkpoint file
    model = init_model(cfg, args.checkpoint, device=args.device)
    if args.preview_model:
        print(model)
        print('\n Please remove `--preview-model` to get the CAM.')
        sys.exit()

    # apply transform and perpare data
    data, src_img = apply_transforms(args.img, cfg.data.test.pipeline)
    data['img'] = data['img'].unsqueeze(0)

    # build target layers
    target_layers = [
        get_layer(layer_str, model) for layer_str in args.target_layers
    ]
    assert len(args.target_layers) != 0, '`--target-layers` can not be empty'

    # init a cam grad calculator
    use_cuda = True if 'cuda' in args.device else False
    reshape_transform = build_reshape_transform(model)
    cam = init_cam(args.method, model, target_layers, use_cuda,
                   reshape_transform)

    # calculate cam grads and show|save the visualization image
    grayscale_cam = cam(
        input_tensor=data['img'],
        target_category=args.target_category,
        eigen_smooth=args.eigen_smooth,
        aug_smooth=args.aug_smooth)
    show_cam_grad(
        grayscale_cam, src_img, title=args.method, out_path=args.save_path)


if __name__ == '__main__':
    main()

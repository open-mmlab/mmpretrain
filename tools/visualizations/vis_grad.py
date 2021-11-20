# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy

import cv2
import torch
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import show_cam_on_image
from mmcv import Config, DictAction

from mmcls.apis import init_model
from mmcls.datasets.pipelines import Compose

FORMAT_TRANSFORMERS_SET = {'ToTensor', 'Normalize', 'ImageToTensor', 'Collect'}

GradCAM_MAP = dict(
    GradCAM=GradCAM, 
    ScoreCAM=ScoreCAM, 
    GradCAMPlusPlus=GradCAMPlusPlus, 
    AblationCAM=AblationCAM, 
    XGradCAM=XGradCAM, 
    EigenCAM=EigenCAM
)

class mmActivationsAndGradients(ActivationsAndGradients):
    def __call__(self, x):
        self.gradients = []
        self.activations = []

        return self.model(x, return_loss=False, return_score=True)

def custom_get_loss(self, output, target_category):
    loss = 0
    for i in range(len(target_category)):
        label = torch.tensor([target_category[i]])
        loss += self.model.head.loss(output, label)['loss']
    return loss

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Cam Grad')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default=None, help='Device to use')
    parser.add_argument('--target_categorys', 
        default=[],
        nargs='+',
        type=int,
        help='Device to use')
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

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def apply_transforms(img_path, pipeline_cfg):
    '''Since there are some transforms, which will change the regin to inference
    such as CenterCrop. So it is necessary to get inference image.
    '''
    data = dict(img_info=dict(filename=img_path), img_prefix=None)

    def split_pipeline_cfg(pipeline_cfg):
        '''to split the transfoms into image_transforms and format_transforms'''
        image_transforms_cfg, format_transforms_cfg = [], []
        if pipeline_cfg[0]['type'] != 'LoadImageFromFile':
            pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
        for transform in pipeline_cfg:
            if transform['type'] in FORMAT_TRANSFORMERS_SET:
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
    # Construct the CAM object once, and then re-use it on many images:
 
    GradCAM_Class = GradCAM_MAP[cam_type]
    MMGradCAM = type('MMGradCAM', (GradCAM_Class, ), {'get_loss':custom_get_loss})
    cam = MMGradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam.activations_and_grads = mmActivationsAndGradients(cam.model, 
                                    cam.target_layers, cam.reshape_transform)
    
    return cam


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the model from a config file and a checkpoint file
    model = init_model(cfg, args.checkpoint, device=args.device)

    # apply transform and perpare data
    data, src_img = apply_transforms(args.img, cfg.data.test.pipeline)
    data['img'] = data['img'].unsqueeze(0)

    # build target layers
    target_layers = [eval('model.backbone.layer4')]

    # init a cam grad computer
    target_category = None
    cam_type = 'GradCAM'
    use_cuda = False
    cam = init_cam(cam_type, model, target_layers, use_cuda)

    # caculate cam grads
    grayscale_cam = cam(input_tensor=data['img'], target_category=target_category)

    # fuse src_img and grayscale_cam 
    grayscale_cam = grayscale_cam[0, :]
    src_img = np.float32(src_img[:, :, ::-1]) / 255
    visualization = show_cam_on_image(src_img, grayscale_cam, use_rgb=True)
    cv2.imshow("cam_type", visualization)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv import Config
from mmdet.apis import inference_detector
from mmdet.models import build_detector

from mmcls.models import (MobileNetV2, MobileNetV3, RegNet, ResNeSt, ResNet,
                          ResNeXt, SEResNet, SEResNeXt, SwinTransformer,
                          TIMMBackbone)
from mmcls.models.backbones.timm_backbone import timm

backbone_configs = dict(
    mobilenetv2=dict(
        backbone=dict(
            type='mmcls.MobileNetV2',
            widen_factor=1.0,
            norm_cfg=dict(type='GN', num_groups=2, requires_grad=True),
            out_indices=(4, 7)),
        out_channels=[96, 1280]),
    mobilenetv3=dict(
        backbone=dict(
            type='mmcls.MobileNetV3',
            norm_cfg=dict(type='GN', num_groups=2, requires_grad=True),
            out_indices=range(7, 12)),
        out_channels=[48, 48, 96, 96, 96]),
    regnet=dict(
        backbone=dict(type='mmcls.RegNet', arch='regnetx_400mf'),
        out_channels=384),
    resnext=dict(
        backbone=dict(
            type='mmcls.ResNeXt', depth=50, groups=32, width_per_group=4),
        out_channels=2048),
    resnet=dict(
        backbone=dict(type='mmcls.ResNet', depth=50), out_channels=2048),
    seresnet=dict(
        backbone=dict(type='mmcls.SEResNet', depth=50), out_channels=2048),
    seresnext=dict(
        backbone=dict(
            type='mmcls.SEResNeXt', depth=50, groups=32, width_per_group=4),
        out_channels=2048),
    resnest=dict(
        backbone=dict(
            type='mmcls.ResNeSt',
            depth=50,
            radix=2,
            reduction_factor=4,
            out_indices=(0, 1, 2, 3)),
        out_channels=[256, 512, 1024, 2048]),
    swin=dict(
        backbone=dict(
            type='mmcls.SwinTransformer',
            arch='small',
            drop_path_rate=0.2,
            img_size=800,
            out_indices=(2, 3)),
        out_channels=[384, 768]),
    timm_efficientnet=dict(
        backbone=dict(
            type='mmcls.TIMMBackbone',
            model_name='efficientnet_b1',
            features_only=True,
            pretrained=False,
            out_indices=(1, 2, 3, 4)),
        out_channels=[24, 40, 112, 320]),
    timm_resnet=dict(
        backbone=dict(
            type='mmcls.TIMMBackbone',
            model_name='resnet50',
            features_only=True,
            pretrained=False,
            out_indices=(1, 2, 3, 4)),
        out_channels=[256, 512, 1024, 2048]))

module_mapping = {
    'mobilenetv2': MobileNetV2,
    'mobilenetv3': MobileNetV3,
    'regnet': RegNet,
    'resnext': ResNeXt,
    'resnet': ResNet,
    'seresnext': SEResNeXt,
    'seresnet': SEResNet,
    'resnest': ResNeSt,
    'swin': SwinTransformer,
    'timm_efficientnet': TIMMBackbone,
    'timm_resnet': TIMMBackbone
}


def test_mmdet_inference():
    config_path = './tests/data/retinanet.py'
    rng = np.random.RandomState(0)
    img1 = rng.rand(100, 100, 3)

    for module_name, backbone_config in backbone_configs.items():
        module = module_mapping[module_name]
        if module is TIMMBackbone and timm is None:
            print(f'skip {module_name} because timm is not available')
            continue
        print(f'test {module_name}')
        config = Config.fromfile(config_path)
        config.model.backbone = backbone_config['backbone']
        out_channels = backbone_config['out_channels']
        if isinstance(out_channels, int):
            config.model.neck = None
            config.model.bbox_head.in_channels = out_channels
            anchor_generator = config.model.bbox_head.anchor_generator
            anchor_generator.strides = anchor_generator.strides[:1]
        else:
            config.model.neck.in_channels = out_channels

        model = build_detector(config.model)
        assert isinstance(model.backbone, module)

        model.cfg = config

        model.eval()
        result = inference_detector(model, img1)
        assert len(result) == config.num_classes

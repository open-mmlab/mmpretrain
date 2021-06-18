from mmdet.models import build_detector

from mmcls.models import (MobileNetV2, MobileNetV3, RegNet, ResNeSt, ResNet,
                          ResNeXt, SEResNet, SEResNeXt, SwinTransformer)

backbone_configs = dict(
    mobilenetv2=dict(
        backbone=dict(
            type='mmcls.MobileNetV2',
            widen_factor=1.0,
            norm_cfg=dict(type='GN', num_groups=2, requires_grad=True),
            out_indices=(4, 7))),
    mobilenetv3=dict(
        backbone=dict(
            type='mmcls.MobileNetV3',
            norm_cfg=dict(type='GN', num_groups=2, requires_grad=True),
            out_indices=range(7, 12))),
    regnet=dict(backbone=dict(type='mmcls.RegNet', arch='regnetx_400mf')),
    resnext=dict(
        backbone=dict(
            type='mmcls.ResNeXt', depth=50, groups=32, width_per_group=4)),
    resnet=dict(backbone=dict(type='mmcls.ResNet', depth=50)),
    seresnet=dict(backbone=dict(type='mmcls.SEResNet', depth=50)),
    seresnext=dict(
        backbone=dict(
            type='mmcls.SEResNeXt', depth=50, groups=32, width_per_group=4)),
    resnest=dict(
        backbone=dict(
            type='mmcls.ResNeSt',
            depth=50,
            radix=2,
            reduction_factor=4,
            out_indices=(0, 1, 2, 3))),
    swin=dict(
        backbone=dict(
            type='mmcls.SwinTransformer', arch='small', drop_path_rate=0.2)))

module_mapping = {
    'mobilenetv2': MobileNetV2,
    'mobilenetv3': MobileNetV3,
    'regnet': RegNet,
    'resnext': ResNeXt,
    'resnet': ResNet,
    'seresnext': SEResNeXt,
    'seresnet': SEResNet,
    'resnest': ResNeSt,
    'swin': SwinTransformer
}


def test_mmdet_inference():
    from mmcv import Config
    config_path = './tests/data/retinanet.py'
    config = Config.fromfile(config_path)

    for module_name, backbone_config in backbone_configs.items():
        config.model.backbone = backbone_config['backbone']
        model = build_detector(config.model)
        module = module_mapping[module_name]
        assert isinstance(model.backbone, module)

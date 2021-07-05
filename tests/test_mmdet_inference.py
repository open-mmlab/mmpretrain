from mmdet import build_detector

backbone_configs = dict(
    mobilenetv2=dict(
        backbone=dict(
            _delete_=True,
            type='mmcv.MobileNetV2',
            widen_factor=1.0,
            norm_cfg=dict(type='GN', num_groups=2, requires_grad=True),
            out_indices=range(3, 7))),
    mobilenetv3=dict(
        backbone=dict(
            _delete_=True,
            type='mmcv.MobileNetV3',
            widen_factor=1.0,
            norm_cfg=dict(type='GN', num_groups=2, requires_grad=True),
            out_indices=range(7, 12))),
    regnet=dict(
        backbone=dict(
            _delete_=True, type='mmcv.RegNet', arch_name='regnetx_400mf')),
    resnext=dict(
        backbone=dict(
            _delete_=True,
            type='mmcv.ResNeXt',
            depth=50,
            groups=32,
            base_width=4)),
    resnet=dict(backbone=dict(_delete_=True, type='mmcv.ResNet', depth=50)),
    seresnet=dict(
        backbone=dict(_delete_=True, type='mmcv.SEResNet', depth=50)),
    seresnext=dict(
        backbone=dict(
            _delete_=True,
            type='mmcv.SEResNeXt',
            depth=50,
            groups=32,
            base_width=4)),
    resnest=dict(
        backbone=dict(
            _delete_=True,
            type='mmcv.ResNeSt',
            depth=50,
            radix=2,
            reduction_factor=4,
            out_indices=(0, 1, 2, 3))),
    swin=dict(
        backbone=dict(
            _delete_=True,
            type='mmcv.SwinTransformer',
            arch='small',
            drop_path_rate=0.2)))


def test_mmdet_inference():
    from mmcv import Config
    config_path = './test/data/retinanet.py'
    config = Config.fromfile(config_path)

    for backbone_config in backbone_configs.items():
        config.merge_from_dict(backbone_config)
        build_detector(config)

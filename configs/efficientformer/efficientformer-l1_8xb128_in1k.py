_base_ = [
    '../_base_/datasets/imagenet_bs128_poolformer_small_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientFormer',
        arch='l1',
        drop_path_rate=0,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['GroupNorm'], val=1., bias=0.),
            dict(type='Constant', layer=['LayerScale'], val=1e-5)
        ]),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='EfficientFormerClsHead', in_channels=448, num_classes=1000))

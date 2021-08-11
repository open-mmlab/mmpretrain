_base_ = [
    '../_base_/models/repvgg.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py',
]
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
model = dict(
    backbone=dict(
            type='RepVGG',
            num_blocks=[4, 6, 16, 1],
            width_multiplier=[2.5, 2.5, 2.5, 5],
            override_groups_map={l: 4 for l in optional_groupwise_layers},
            deploy=False),
    neck=None,
    head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=2560,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
    ))
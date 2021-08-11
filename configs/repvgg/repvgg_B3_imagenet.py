_base_ = [
    '../_base_/models/repvgg.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py',
]
model = dict(
    backbone=dict(
            type='RepVGG',
            num_blocks=[4, 6, 16, 1],
            width_multiplier=[3, 3, 3, 5],
            override_groups_map=None,
            deploy=False),
    neck=None,
    head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=2560,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
    ))
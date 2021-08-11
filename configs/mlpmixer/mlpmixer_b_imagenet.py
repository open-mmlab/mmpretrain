_base_ = [
    '../_base_/models/mlpmixer.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py',
]
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MLPMixer',
        in_channels=3,
        image_size=224,
        patch_size=16,
        dim=768,
        num_layers=12,
        token_dim=384,
        channel_dim=3072
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1),
    ))
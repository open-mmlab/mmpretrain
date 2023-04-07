_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MaxViT',
        embed_dim=(128, 256, 512, 1024),
        depths=(2, 6, 14, 2),
        img_size=224,
        stem_width=128,
        head_hidden_size=1024,
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

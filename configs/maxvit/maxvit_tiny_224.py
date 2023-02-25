_base_ = [
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MaxViT',
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        stem_width=64,
        head_hidden_size=512,  # is head in_channels
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

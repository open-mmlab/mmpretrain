_base_ = [
    '../_base_/datasets/imagenet_bs64_maxvit_512.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# dataset settings
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MaxViT',
        embed_dim=(192, 384, 768, 1536),
        depths=(2, 6, 14, 2),
        img_size=512,
        stem_width=192,
        head_hidden_size=1536,
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

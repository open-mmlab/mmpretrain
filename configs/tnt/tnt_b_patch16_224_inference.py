_base_ = [
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TNT',
        arch='b',
        img_size=224,
        patch_size=16,
        in_channels=3,
        ffn_ratio=4,
        qkv_bias=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        first_stride=4,
        num_fcs=2,
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=.02),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=640,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1, 5),
        init_cfg=dict(type='TruncNormal', layer='Linear', std=.02)),
    train_cfg=dict(mixup=dict(alpha=0.2, num_classes=1000)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(248, -1),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

dataset_type = 'ImageNet'
data = dict(
    samples_per_gpu=32, workers_per_gpu=4, test=dict(pipeline=test_pipeline))

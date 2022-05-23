_base_ = [
    '../_base_/models/repmlp-base_224.py',
    '../_base_/datasets/imagenet_bs64_mixer_224.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

default_hooks = dict(optimizer=dict(grad_clip=dict(max_norm=1.0)))

model = dict(backbone=dict(img_size=256))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale_factor=(256 * 256 // 224, -1),
        keep_ratio=True,
        backend='pillow'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))

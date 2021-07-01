_base_ = [
    '../_base_/models/tnt_s_patch16_224.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

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

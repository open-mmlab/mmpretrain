# accuracy_top-1 : 81.52 accuracy_top-5 : 95.73
_base_ = [
    '../_base_/models/tnt_s_patch16_224.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(248, -1),
        keep_ratio=True,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

dataset_type = 'ImageNet'
data = dict(
    samples_per_gpu=64, workers_per_gpu=4, test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=295, by_epoch=True, begin=5, end=300)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300)
val_cfg = dict(interval=1)  # validate every epoch
test_cfg = dict()

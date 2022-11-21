_base_ = [
    '../_base_/models/mobilenet_v3/mobilenet_v3_small_075_imagenet.py',
    '../_base_/datasets/imagenet_bs128_mbv3.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(backbone=dict(norm_cfg=dict(type='BN', eps=1e-5, momentum=0.1)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies='imagenet',
        hparams=dict(pad_val=[round(x) for x in [103.53, 116.28, 123.675]])),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        type='RMSprop',
        lr=0.064,
        alpha=0.9,
        momentum=0.9,
        eps=0.0316,
        weight_decay=1e-5))

param_scheduler = dict(type='StepLR', by_epoch=True, step_size=2, gamma=0.973)

train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)

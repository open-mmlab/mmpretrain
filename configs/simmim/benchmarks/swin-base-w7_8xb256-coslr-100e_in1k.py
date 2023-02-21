_base_ = 'swin-base-w6_8xb256-coslr-100e_in1k-192px.py'

# model
model = dict(
    backbone=dict(
        img_size=224, stage_cfgs=dict(block_cfgs=dict(window_size=7))))

# train pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackClsInputs')
]

# test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline),
    collate_fn=dict(type='default_collate'),
    pin_memory=True)
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline),
    collate_fn=dict(type='default_collate'),
    pin_memory=True)
test_dataloader = val_dataloader

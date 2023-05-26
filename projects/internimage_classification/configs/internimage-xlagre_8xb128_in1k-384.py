_base_ = './_base_.py'

model = dict(
    backbone=dict(
        stem_channels=192,
        drop_path_rate=0.2,
        stage_blocks=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        layer_scale=1e-5,
        offset_scale=2.0,
        post_norm=True),
    head=dict(in_channels=2304))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=384,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='PackInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(lr=5e-6))
param_scheduler = [
    dict(
        type='LinearLR',
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=18, by_epoch=True, begin=2, end=20)
]
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)

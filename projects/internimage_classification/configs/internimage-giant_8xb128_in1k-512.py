_base_ = './_base_.py'

model = dict(
    backbone=dict(
        stem_channels=512,
        drop_path_rate=0.4,
        stage_blocks=[2, 2, 48, 4],
        groups=[16, 32, 64, 128],
        dw_kernel_size=5,
        level2_post_norm=True,
        level2_post_norm_block_ids=[5, 11, 17, 23, 29, 35, 41, 47],
        center_feature_scale=True,
        use_clip_projector=True,
    ),
    neck=None,
    head=dict(in_channels=768))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=512,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=512,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=512),
    dict(type='PackInputs'),
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

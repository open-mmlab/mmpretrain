_base_ = './_base_.py'

model = dict(
    backbone=dict(
        stem_channels=320,
        drop_path_rate=0.1,
        stage_blocks=[6, 6, 32, 6],
        groups=[10, 20, 40, 80],
        dw_kernel_size=5,
        res_post_norm=True,
        level2_post_norm=True,
        level2_post_norm_block_ids=[5, 11, 17, 23, 29],
        center_feature_scale=True,
        use_clip_projector=True,
    ),
    neck=None,
    head=dict(in_channels=768))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=640,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=640,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=640),
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

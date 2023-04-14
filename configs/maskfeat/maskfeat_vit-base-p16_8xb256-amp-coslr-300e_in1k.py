_base_ = '../_base_/default_runtime.py'

# dataset settings
dataset_type = 'ImageNet'
data_root = 'data/imagenet/'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.5, 1.0),
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='BEiTMaskGenerator',
        input_size=14,
        num_masking_patches=78,
        min_num_patches=15,
    ),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))

# model settings
model = dict(
    type='MaskFeat',
    backbone=dict(type='MaskFeatViT', arch='b', patch_size=16),
    neck=dict(
        type='LinearNeck',
        in_channels=768,
        out_channels=108,
        norm_cfg=None,
        init_cfg=dict(type='TruncNormal', layer='Linear', std=0.02, bias=0)),
    head=dict(
        type='MIMHead',
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    target_generator=dict(
        type='HOGGenerator', nbins=9, pool=8, gaussian_window=16))

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=2e-4 * 8, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=0.02),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            # 'pos_embed': dict(decay_mult=0.),
            # 'cls_token': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=30,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=270,
        eta_min=1e-6,
        by_epoch=True,
        begin=30,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)

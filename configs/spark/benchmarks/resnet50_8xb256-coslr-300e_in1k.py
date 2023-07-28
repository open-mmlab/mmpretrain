_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/datasets/imagenet_bs256_rsb_a12.py',
    '../../_base_/default_runtime.py'
]
# modification is based on ResNets RSB settings
data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='NumpyToPIL', to_rgb=True),
    dict(
        type='torchvision/TrivialAugmentWide',
        num_magnitude_bins=31,
        interpolation='bicubic',
        fill=None),
    dict(type='PILToNumpy', to_bgr=True),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# model settings
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        drop_path_rate=0.05,
        init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.')),
    head=dict(
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, use_sigmoid=True)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1),
        dict(type='CutMix', alpha=1.0)
    ]))

# schedule settings
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='Lamb',
        lr=0.016,
        weight_decay=0.02,
    ),
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.7,
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=300)
]
train_cfg = dict(by_epoch=True, max_epochs=300)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    # only keeps the latest 2 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))
# randomness
randomness = dict(seed=0, diff_rank_seed=True)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)

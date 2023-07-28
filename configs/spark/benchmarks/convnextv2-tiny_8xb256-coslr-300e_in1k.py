_base_ = [
    '../../_base_/datasets/imagenet_bs64_swin_224.py',
    '../../_base_/default_runtime.py',
]

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

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline),
    sampler=dict(type='RepeatAugSampler', shuffle=True),
)

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        drop_path_rate=0.1,
        layer_scale_init_value=0.,
        use_grn=True,
        init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.')),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=dict(type='TruncNormal', layer='Linear', std=.02, bias=0.),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]),
)

custom_hooks = [
    dict(
        type='EMAHook',
        momentum=1e-4,
        evaluate_on_origin=True,
        priority='ABOVE_NORMAL')
]

# schedule settings
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=3.2e-3, betas=(0.9, 0.999), weight_decay=0.05),
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
        end=20,
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=280,
        eta_min=1.0e-5,
        by_epoch=True,
        begin=20,
        end=300)
]
train_cfg = dict(by_epoch=True, max_epochs=300)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    # only keeps the latest 2 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)

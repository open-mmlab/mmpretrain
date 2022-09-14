_base_ = ['../_base_/default_runtime.py']

# dataset settings
dataset_type = 'ImageNet'
data_preprocessor = dict(
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
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=15,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='RepeatAugSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=256,
    num_workers=15,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.001,
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
        end=300),
    dict(
        type='LinearParamScheduler',
        param_name='weight_decay',
        start_factor=1,
        end_factor=0.1,
        by_epoch=True,
        begin=0,
        end=300)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1024)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s0',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
        ),
        topk=(1, 5),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1, num_classes=1000),
        dict(type='CutMix', alpha=1.0, num_classes=1000)
    ]))

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=5e-4, priority='ABOVE_NORMAL')]

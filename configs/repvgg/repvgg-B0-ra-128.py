_base_ = [
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RepVGG',
        arch='B0',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2560,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            num_classes=1000),
        topk=(1, 5),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)

# schedule settings
optim_wrapper = dict(
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys={
            'branch_3x3.norm': dict(decay_mult=0.0),
            'branch_1x1.norm': dict(decay_mult=0.0),
            'branch_norm.bias': dict(decay_mult=0.0),
        }))

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
    dict(type='RandomResizedCrop', scale=192),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=5,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=232, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=12,
    dataset=dict(pipeline=train_pipeline),
    sampler=dict(type='RepeatAugSampler', shuffle=True))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# schedule settings
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

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# dataset settings
dataset_type = 'CustomDataset'

# ImageNet pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=224,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='KFoldDataset',
        num_splits=5,
        fold=0,
        seed=1234,
        dataset=dict(
            type=dataset_type,
            pipeline=train_pipeline,
            data_prefix='data/food_dataset',
        )),
    val=dict(
        type='KFoldDataset',
        num_splits=5,
        fold=0,
        seed=1234,
        test_mode=True,
        dataset=dict(
            type=dataset_type,
            pipeline=test_pipeline,
            data_prefix='data/food_dataset',
        )),
    test=dict(
        type='KFoldDataset',
        num_splits=5,
        fold=0,
        seed=1234,
        test_mode=True,
        dataset=dict(
            type=dataset_type,
            pipeline=test_pipeline,
            data_prefix='data/food_dataset',
        )),
    )

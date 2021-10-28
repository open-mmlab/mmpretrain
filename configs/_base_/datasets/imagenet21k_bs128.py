# dataset settings
dataset_type = 'ImageNet21k'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet21k/train',
        pipeline=train_pipeline,
        recursion_subdir=True),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet21k/val',
        ann_file='data/imagenet21k/meta/val.txt',
        pipeline=test_pipeline,
        recursion_subdir=True),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet21k/val',
        ann_file='data/imagenet21k/meta/val.txt',
        pipeline=test_pipeline,
        recursion_subdir=True))
evaluation = dict(interval=1, metric='accuracy')

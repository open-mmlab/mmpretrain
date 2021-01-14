_base_ = ['./_base_/default_runtime.py']

# dataset
dataset_type = 'VOC'
# how to obtain this?
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='RandomResizedCrop', size=224),
    # some paper use 448 for some reason?
    dict(type='Resize', size=(224, 224)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/VOCdevkit/VOC2007/',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/VOCdevkit/VOC2007/',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/VOCdevkit/VOC2007/',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metric=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'OF1'])

# model

model = dict(
    type='ImageClassifier',
    backbone=dict(type='VGG', depth=16, num_classes=20),
    neck=None,
    head=dict(
        type='MultiLabelClsHead',
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

# schedules
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

load_from = 'https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_imagenet-91b6d117.pth'  # noqa

# optimizer
optimizer = dict(type='SGD', lr=0.001 / 8, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[20, 40, 60, 80])
runner = dict(type='EpochBasedRunner', max_epochs=100)

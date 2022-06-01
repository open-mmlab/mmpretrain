# dataset settings
dataset_type = 'VOC'
preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/VOCdevkit/VOC2007/',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/VOCdevkit/VOC2007/',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metric=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'OF1'])

train_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/VOCdevkit/VOC2007/',
        # manually split the `trainval.txt` for standard training.
        ann_file='ImageSets/Main/trainval.txt',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/VOCdevkit/VOC2007/',
        # manually split the `trainval.txt` for standard validation.
        ann_file='ImageSets/Main/test.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(
    type='MultiLabelMetric',
    items=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'OF1'])

test_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/VOCdevkit/VOC2007/',
        ann_file='ImageSets/Main/test.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
test_evaluator = val_evaluator

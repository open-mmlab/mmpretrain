# dataset settings
dataset_type = 'CUB'
preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=600),
    dict(type='RandomCrop', size=448),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=600),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackClsInputs'),
]

common_data_cfg = dict(
    type=dataset_type,
    data_root='data/CUB_200_2011',
    ann_file='images.txt',
    image_class_labels_file='image_class_labels.txt',
    train_test_split_file='train_test_split.txt',
    data_prefix='images',
)

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(**common_data_cfg, test_mode=False, pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(**common_data_cfg, test_mode=True, pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

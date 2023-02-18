# dataset settings
dataset_type = 'NIHChestXRays'
data_preprocessor = dict(
    num_classes=14,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
    # generate onehot-format labels for multi-label classification.
    to_onehot=True,
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

tta_model = dict(type='AverageClsScoreTTA')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(
        type='TenCropTestTimeAug',
        size=224,
        transforms=dict(type='PackClsInputs')),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/nih-chestxrays',
        ann_file='Data_Entry_2017.csv',
        train_test_split_file='train_val_list.txt',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/nih-chestxrays',
        ann_file='Data_Entry_2017.csv',
        train_test_split_file='test_list.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/nih-chestxrays',
        ann_file='Data_Entry_2017.csv',
        train_test_split_file='test_list.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = [
    dict(type='ROCAUC'),
    dict(type='ROCAUC', classwise=True),
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

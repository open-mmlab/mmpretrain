# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=12,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448)),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/transformer/cla/DAGM/train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/transformer/cla/DAGM/test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = [
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score'])
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

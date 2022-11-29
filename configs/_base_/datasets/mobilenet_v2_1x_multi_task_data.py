# dataset settings
dataset_type = 'MultiTaskDataset'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 200), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatMultiTaskLabels'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 200), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatMultiTaskLabels')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        ann_file='data/MEDIC_train.json',
        data_root='../dataset',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        ann_file='data/MEDIC_test.json',
        data_root='../dataset',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)
test_dataloader = dict(
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        ann_file='data/MEDIC_test.json',
        data_root='../dataset',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_evaluator = dict(
    type='MultiTasksMetric',
    task_metrics={
        'damage_severity': [dict(type='Accuracy', topk=(1, ))],
        'informative': [dict(type='Accuracy', topk=(1, ))],
        'disaster_types': [dict(type='Accuracy', topk=(1, ))],
        'humanitarian': [dict(type='Accuracy', topk=(1, ))]
    })

test_dataloader = val_dataloader
test_evaluator = val_evaluator

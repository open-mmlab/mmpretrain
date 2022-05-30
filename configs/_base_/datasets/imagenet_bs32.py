# dataset settings
dataset_type = 'ImageNet'

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/classification/',
#         'data/': 's3://openmmlab/datasets/classification/'
#     }))

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(256, -1), keep_ratio=True),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

evaluation = dict(interval=1, metric='accuracy')

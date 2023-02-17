# dataset settings
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

view_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomGrayscale', prob=0.2, keep_channels=True),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))

# dataset settings
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.5, 1.0),
        ratio=(0.75, 1.3333),
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='BEiTMaskGenerator',
        input_size=14,
        num_masking_patches=78,
        min_num_patches=15,
    ),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))

# for visualization
vis_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(224, 224), backend='pillow'),
    dict(
        type='BEiTMaskGenerator',
        input_size=14,
        num_masking_patches=78,
        min_num_patches=15,
    ),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]

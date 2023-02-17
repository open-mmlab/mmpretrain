# dataset settings
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResizedCropAndInterpolationWithTwoPic',
        size=224,
        second_size=112,
        interpolation='bicubic',
        second_interpolation='lanczos',
        scale=(0.08, 1.0)),
    dict(
        type='BEiTMaskGenerator',
        input_size=(14, 14),
        num_masking_patches=75,
        max_num_patches=None,
        min_num_patches=16),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=256,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))

# dataset settings
dataset_type = 'ImageNet'
data_root = 'data/imagenet/'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=192,
        scale=(0.67, 1.0),
        ratio=(3. / 4., 4. / 3.)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='SimMIMMaskGenerator',
        input_size=192,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
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
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(192, 192), backend='pillow'),
    dict(
        type='SimMIMMaskGenerator',
        input_size=192,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]

_base_ = ['./pipelines/rand_aug.py']

# Dataset settings
dataset_type = 'ImageNet'

# Meta infos
# The loaded images are in BGR format before Normalize
img_mean_rgb = [123.675, 116.28, 103.53]
img_mean_bgr = img_mean_rgb[::-1]
img_std_rgb = [58.395, 57.12, 57.375]
img_std_bgr = img_std_rgb[::-1]
img_norm_cfg = dict(mean=img_mean_rgb, std=img_std_rgb, to_rgb=True)
resize_cfg = dict(backend='pillow', interpolation='bicubic')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, **resize_cfg),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in img_mean_bgr], **resize_cfg),
    ),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_mean_bgr,
        fill_std=img_std_bgr,
    ),
    dict(type='Normalize', mean=img_mean_rgb, std=img_std_rgb, to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), **resize_cfg),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', mean=img_mean_rgb, std=img_std_rgb, to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='accuracy', save_best='auto')

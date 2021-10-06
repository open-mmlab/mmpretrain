data_root = '/home/ubuntu/data/'

policies = [
    [
        dict(type='Posterize', bits=4, prob=0.4),
        dict(type='Rotate', angle=30., prob=0.6)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.6)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Equalize', prob=0.6)],
    [
        dict(type='Posterize', bits=5, prob=0.6),
        dict(type='Posterize', bits=5, prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 6, prob=0.6),
        dict(type='Equalize', prob=0.6)
    ],
    [dict(type='Posterize', bits=6, prob=0.8),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='Rotate', angle=10., prob=0.2),
        dict(type='Solarize', thr=256 / 9, prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.6),
        dict(type='Posterize', bits=5, prob=0.4)
    ],
    [
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),
        dict(type='ColorTransform', magnitude=0., prob=0.4)
    ],
    [
        dict(type='Rotate', angle=30., prob=0.4),
        dict(type='Equalize', prob=0.6)
    ],
    [dict(type='Equalize', prob=0.0),
     dict(type='Equalize', prob=0.8)],
    [dict(type='Invert', prob=0.6),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
        dict(type='Contrast', magnitude=0.8, prob=1.)
    ],
    [
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),
        dict(type='ColorTransform', magnitude=0.2, prob=1.)
    ],
    [
        dict(type='ColorTransform', magnitude=0.8, prob=0.8),
        dict(type='Solarize', thr=256 / 9 * 2, prob=0.8)
    ],
    [
        dict(type='Sharpness', magnitude=0.7, prob=0.4),
        dict(type='Invert', prob=0.6)
    ],
    [
        dict(
            type='Shear',
            magnitude=0.3 / 9 * 5,
            prob=0.6,
            direction='horizontal'),
        dict(type='Equalize', prob=1.)
    ],
    [
        dict(type='ColorTransform', magnitude=0., prob=0.4),
        dict(type='Equalize', prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.6)
    ],
    [dict(type='Invert', prob=0.6),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
        dict(type='Contrast', magnitude=0.8, prob=1.)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Equalize', prob=0.6)],
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CropBoundingBox'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='AutoAugment', policies=policies),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='const',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SwapChannels'),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CropBoundingBox'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SwapChannels'),
    dict(type='Collect', keys=['img']),
]

all_classes = data_root + 'all_classes.txt'
ob_root = data_root + 'OpenBrands/'
train_openbrand = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='OpenBrandDataset',
            classes=all_classes,
            ann_file=ob_root + 'annotations/train_20210409_14_reduced.json',
            data_prefix=ob_root + '电商标识检测大赛_train_20210409_14/',
            pipeline=train_pipeline,
        ),
        dict(
            type='OpenBrandDataset',
            classes=all_classes,
            ann_file=ob_root + 'annotations/train_20210409_15_reduced.json',
            data_prefix=ob_root + '电商标识检测大赛_train_20210409_15/',
            pipeline=train_pipeline,
        ),
    ],
)

validation_openbrand=dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='OpenBrandDataset',
            classes=all_classes,
            ann_file=ob_root + 'annotations/validation_20210409_14_reduced.json',
            data_prefix=ob_root + '电商标识检测大赛_train_20210409_14/',
            pipeline=test_pipeline,
        ),
        dict(
            type='OpenBrandDataset',
            classes=all_classes,
            ann_file=ob_root + 'annotations/validation_20210409_15_reduced.json',
            data_prefix=ob_root + '电商标识检测大赛_train_20210409_15/',
            pipeline=test_pipeline,
        ),
    ],
)

test_openbrand = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='OpenBrandDataset',
            classes=all_classes,
            ann_file=ob_root + 'annotations/test_20210409_14_reduced.json',
            data_prefix=ob_root + '电商标识检测大赛_train_20210409_14/',
            pipeline=test_pipeline,
        ),
        dict(
            type='OpenBrandDataset',
            classes=all_classes,
            ann_file=ob_root + 'annotations/test_20210409_15_reduced.json',
            data_prefix=ob_root + '电商标识检测大赛_train_20210409_15/',
            pipeline=test_pipeline,
        ),
    ],
)

ld_root = data_root + 'LogoDet-3K/'
rp_root = data_root + 'logo_dataset/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=[
        dict(
            type='XMLDataset',
            classes=all_classes,
            ann_file=ld_root + 'train_micro.txt',
            ann_subdir='',
            data_prefix=ld_root,
            img_subdir='',
            pipeline=train_pipeline,
        ),
        dict(
            type='XMLDataset',
            classes=all_classes,
            ann_file=rp_root + 'ImageSets/Main/train_micro.txt',
            ann_subdir='Annotations',
            data_prefix=rp_root,
            img_subdir='JPEGImages',
            pipeline=train_pipeline,
        ),
        train_openbrand,
    ],
    val=[
        dict(
            type='XMLDataset',
            classes=all_classes,
            ann_file=ld_root + 'val_micro.txt',
            ann_subdir='',
            data_prefix=ld_root,
            img_subdir='',
            pipeline=test_pipeline,
        ),
        dict(
            type='XMLDataset',
            classes=all_classes,
            ann_file=rp_root + 'ImageSets/Main/validation_micro.txt',
            ann_subdir='Annotations',
            data_prefix=rp_root,
            img_subdir='JPEGImages',
            pipeline=test_pipeline,
        ),
        validation_openbrand,
    ],
    test=[
        dict(
            type='XMLDataset',
            classes=all_classes,
            ann_file=ld_root + 'test_micro.txt',
            ann_subdir='',
            data_prefix=ld_root,
            img_subdir='',
            pipeline=test_pipeline,
        ),
        dict(
            type='XMLDataset',
            classes=all_classes,
            ann_file=rp_root + 'ImageSets/Main/test_micro.txt',
            ann_subdir='Annotations',
            data_prefix=rp_root,
            img_subdir='JPEGImages',
            pipeline=test_pipeline,
        ),
        test_openbrand,
    ]
)
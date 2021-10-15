data_root = '/home/ubuntu/data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CropBoundingBox'),
    dict(type='Resize', size=(224, 224)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CropBoundingBox'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
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
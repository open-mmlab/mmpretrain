# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=5000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow'),
    # dict(type='RandomResizedCrop', scale=256, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow'),
    # dict(type='ResizeEdge', scale=292, edge='short', backend='pillow'),
    # dict(type='CenterCrop', crop_size=256),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        data_root='data/ACCV_workshop',
        ann_file='meta/all.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        data_root='data/ACCV_workshop',
        ann_file='meta/val.txt',
        data_prefix='train',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = val_evaluator

test_dataloader = val_dataloader

# data settings

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=384,
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='PackInputs',
        algorithm_keys=['text', 'gt_bboxes'],
        meta_keys=['image_id'],
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(480, 480),
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='PackInputs',
        algorithm_keys=['text', 'gt_bboxes'],
        meta_keys=['image_id'],
    ),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RefCOCO',
        data_root='data/coco',
        ann_file='annotations/refcoco_train.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RefCOCO',
        data_root='data/coco',
        ann_file='annotations/refcoco_val.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RefCOCO',
        data_root='data/coco',
        ann_file='annotations/refcoco_test.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# TODO: add visual grounding mAP metrics
val_evaluator = []
test_evaluator = val_evaluator

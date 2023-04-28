# data settings

data_preprocessor = dict(
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
        data_prefix='train2014',
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RefCOCO',
        data_root='data/coco',
        data_prefix='train2014',
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_dataloader = val_dataloader

# TODO: add visual grounding mAP metrics
val_evaluator = []
test_evaluator = val_evaluator

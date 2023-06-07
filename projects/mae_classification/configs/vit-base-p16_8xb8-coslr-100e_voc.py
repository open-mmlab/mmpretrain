_base_ = './_base_.py'

# dataset settings
dataset_type = 'VOC'
num_classes = 20
data_preprocessor = dict(
    num_classes=num_classes,
    to_onehot=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=256),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=5,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124])),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=256),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='PackInputs',
        # `gt_label_difficult` is needed for VOC evaluation
        meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'gt_label_difficult')),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/VOC2007',
        split='trainval',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/VOC2007',
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_dataloader = val_dataloader

# calculate precision_recall_f1 and mAP
val_evaluator = [
    dict(type='VOCMultiLabelMetric'),
    dict(type='VOCMultiLabelMetric', average='micro'),
    dict(type='VOCAveragePrecision')
]
test_evaluator = val_evaluator

# model settings
model = dict(
    head=dict(type='MultiLabelLinearClsHead', num_classes=num_classes))

# optimizer wrapper
optim_wrapper = dict(optimizer=dict(lr=5e-05, weight_decay=0.001), )

_base_ = './_base_.py'

# dataset settings
dataset_type = 'Flowers102'
num_classes = 102
data_preprocessor = dict(num_classes=num_classes)

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
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/Flowers102',
        split='trainval',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/Flowers102',
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# model settings
model = dict(head=dict(num_classes=num_classes))

# optimizer wrapper
optim_wrapper = dict(optimizer=dict(lr=0.0001, weight_decay=0.01), )

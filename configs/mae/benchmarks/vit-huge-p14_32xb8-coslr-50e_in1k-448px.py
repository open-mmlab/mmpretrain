_base_ = 'vit-huge-p16_8xb128-coslr-50e_in1k.py'
# MAE fine-tuning setting

# model settings
# MAE ViT-huge set drop_path_rate to 0.3
model = dict(
    backbone=dict(
        arch='huge', drop_path_rate=0.3, patch_size=14, img_size=448),
    head=dict(in_channels=1280))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=448,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackClsInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=512,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackClsInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# optimizer settings
optim_wrapper = dict(optimizer=dict(lr=0.004, layer_decay_rate=0.75))

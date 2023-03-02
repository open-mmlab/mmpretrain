_base_ = [
    '../_base_/models/resnest50.py',
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py',
    './_randaug_policies.py',
]

# dataset settings

# lighting params, in order of BGR
EIGVAL = [55.4625, 4.7940, 1.1475]
EIGVEC = [
    [-0.5836, -0.6948, 0.4203],
    [-0.5808, -0.0045, -0.8140],
    [-0.5675, 0.7192, 0.4009],
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandAugment',
        policies={{_base_.policies}},
        num_policies=2,
        magnitude_level=12),
    dict(type='EfficientNetRandomCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='Lighting',
        eigval=EIGVAL,
        eigvec=EIGVEC,
        alphastd=0.1,
        to_rgb=False),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=256, backend='pillow'),
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.8, momentum=0.9, weight_decay=1e-4),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=265,
        by_epoch=True,
        begin=5,
        end=270,
    )
]

train_cfg = dict(by_epoch=True, max_epochs=270)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=2048)

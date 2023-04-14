_base_ = [
    '../_base_/datasets/imagenet_bs32_byol.py',
    '../_base_/default_runtime.py',
]
# datasets
train_dataloader = dict(batch_size=256)

# model settings
model = dict(
    type='BarlowTwins',
    backbone=dict(
        type='ResNet',
        depth=50,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=8192,
        out_channels=8192,
        num_layers=3,
        with_last_bn=False,
        with_last_bn_affine=False,
        with_avg_pool=True,
        init_cfg=dict(
            type='Kaiming', distribution='uniform', layer=['Linear'])),
    head=dict(
        type='LatentCrossCorrelationHead',
        in_channels=8192,
        loss=dict(type='CrossCorrelationLoss')))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='LARS', lr=1.6, momentum=0.9, weight_decay=1e-6),
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lr_mult=0.024, lars_exclude=True),
            'bias': dict(decay_mult=0, lr_mult=0.024, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(
                decay_mult=0, lr_mult=0.024, lars_exclude=True),
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.6e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        eta_min=0.0016,
        by_epoch=True,
        begin=10,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)

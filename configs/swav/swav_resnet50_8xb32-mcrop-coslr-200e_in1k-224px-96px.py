_base_ = [
    '../_base_/datasets/imagenet_bs32_swav_mcrop-2-6.py',
    '../_base_/schedules/imagenet_lars_coslr_200e.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='SwAV',
    data_preprocessor=dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='SwAVNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='SwAVHead',
        loss=dict(
            type='SwAVLoss',
            feat_dim=128,  # equal to neck['out_channels']
            epsilon=0.05,
            temperature=0.1,
            num_crops={{_base_.num_crops}},
        )))

# additional hooks
custom_hooks = [
    dict(
        type='SwAVHook',
        priority='VERY_HIGH',
        batch_size={{_base_.train_dataloader.batch_size}},
        epoch_queue_starts=15,
        crops_for_assign=[0, 1],
        feat_dim=128,
        queue_length=3840,
        frozen_layers_cfg=dict(prototypes=5005))
]

# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='LARS', lr=0.6))

find_unused_parameters = True

# learning policy
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        eta_min=6e-4,
        by_epoch=True,
        begin=0,
        end=200,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

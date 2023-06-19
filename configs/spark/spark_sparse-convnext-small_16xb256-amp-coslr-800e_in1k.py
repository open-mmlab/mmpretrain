_base_ = [
    '../_base_/datasets/imagenet_bs512_mae.py',
    '../_base_/default_runtime.py',
]

# dataset 8 x 512
train_dataloader = dict(batch_size=256, num_workers=8)

# model settings
model = dict(
    type='SparK',
    input_size=224,
    downsample_raito=32,
    mask_ratio=0.6,
    enc_dec_norm_cfg=dict(type='SparseLN2d', eps=1e-6),
    enc_dec_norm_dim=768,
    backbone=dict(
        type='SparseConvNeXt',
        arch='small',
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        gap_before_output=False),
    neck=dict(
        type='SparKLightDecoder',
        feature_dim=512,
        upsample_ratio=32,  # equal to downsample_raito
        mid_channels=0,
        last_act=False),
    head=dict(
        type='SparKPretrainHead',
        loss=dict(type='PixelReconstructionLoss', criterion='L2')))

# optimizer wrapper
optimizer = dict(
    type='Lamb', lr=2e-4 * 4096 / 512, betas=(0.9, 0.95), weight_decay=0.04)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            'mask_token': dict(decay_mult=0.),
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=760,
        by_epoch=True,
        begin=40,
        end=800,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingWeightDecay',
        eta_min=0.2,
        T_max=800,
        by_epoch=True,
        begin=0,
        end=800,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)

_base_ = [
    '../_base_/datasets/imagenet_bs128_mixmim.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixMIM',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MixMIMTransformerPretrain',
        arch='B',
        drop_rate=0.0,
        drop_path_rate=0.0,  # drop_path_rate=0.0 during pretraining
    ),
    neck=dict(
        type='MixMIMPretrainDecoder',
        num_patches=49,
        encoder_stride=32,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16),
    head=dict(
        type='MixMIMPretrainHead',
        norm_pix=True,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')))

# optimizer wrapper

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * (2048 / 256),
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={
        'ln': dict(decay_mult=0.0),
        'bias': dict(decay_mult=0.0)
    }))

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
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)

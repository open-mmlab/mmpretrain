_base_ = [
    '../_base_/datasets/imagenet_bs256_cae.py',
    '../_base_/default_runtime.py',
]

# dataset 8GPUs x 256
train_dataloader = dict(batch_size=256, num_workers=16)

# model settings
model = dict(
    type='CAE',
    backbone=dict(
        type='CAEViT',
        arch='b',
        patch_size=16,
        init_values=0.1,
        qkv_bias=False),
    neck=dict(
        type='CAENeck',
        patch_size=16,
        embed_dims=768,
        num_heads=12,
        regressor_depth=4,
        decoder_depth=4,
        mlp_ratio=4,
        init_values=0.1,
    ),
    head=dict(type='CAEHead', loss=dict(type='CAELoss', lambd=2)),
    target_generator=dict(
        type='DALL-E',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/dalle_encoder.pth',  # noqa: E501
        )),
    data_preprocessor=dict(
        type='mmselfsup.CAEDataPreprocessor',
        mean=[124, 117, 104],
        std=[59, 58, 58],
        bgr_to_rgb=True),
    base_momentum=0.0)

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1.5e-3, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=3.0),
    paramwise_cfg=dict(
        bias_decay_mult=0.0, norm_decay_mult=0.0, flat_decay_mult=0.0))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        eta_min=1e-5,
        by_epoch=True,
        begin=10,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
# pre-train for 300 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)

find_unused_parameters = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)

_base_ = [
    '../_base_/datasets/imagenet_cae.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
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
optimizer = dict(type='AdamW', lr=1.5e-3, betas=(0.9, 0.999))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=optimizer,
    clip_grad=dict(max_norm=3.0),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'gamma': dict(decay_mult=0.0)
        }))

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
train_cfg = dict(max_epochs=300)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)

find_unused_parameters = True

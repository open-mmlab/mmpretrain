_base_ = [
    '../_base_/datasets/imagenet_beit.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='BEiT',
    backbone=dict(
        type='BEiTViT',
        arch='base',
        patch_size=16,
        drop_path_rate=0.1,
        final_norm=True,
        layer_scale_init_value=0.1,
        init_cfg=[
            dict(type='TruncNormal', std=0.02, layer='Linear'),
            dict(type='TruncNormal', std=0.02, layer='Conv2d'),
            dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    neck=None,
    head=dict(
        type='BEiTV1Head',
        embed_dims=768,
        num_embed=8192,
        loss=dict(type='BEiTLoss')),
    target_generator=dict(
        type='DALL-E',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/dalle_encoder.pth',  # noqa: E501
        )))

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=1.5e-3, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=3.0),
    paramwise_cfg=dict(
        custom_keys={
            # the following configurations are designed for BEiT
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            'q_bias': dict(decay_mult=0.0),
            'v_bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.gamma': dict(decay_mult=0.0),
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
        eta_min=1e-5,
        by_epoch=True,
        begin=10,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)

find_unused_parameters = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)

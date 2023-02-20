_base_ = [
    '../_base_/models/beitv2_vit-base-p16.py',
    '../_base_/datasets/imagenet_beitv2.py',
    '../_base_/schedules/adamw_coslr-300e_in1k.py',
    '../_base_/default_runtime.py',
]

# optimizer wrapper
# betas: (0.9, 0.98) for 300 epochs and (0.9, 0.999) for 1600 epochs.
optimizer = dict(type='AdamW', lr=1.5e-3, betas=(0.9, 0.98), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=optimizer,
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

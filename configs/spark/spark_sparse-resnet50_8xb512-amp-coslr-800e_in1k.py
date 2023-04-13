_base_ = [
    '../_base_/models/spark_sparse-resnet50.py',
    '../_base_/datasets/imagenet_mae.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset 8 x 512
train_dataloader = dict(batch_size=512, num_workers=8)

# optimizer wrapper
optimizer = dict(
    type='LAMB', lr=2e-4 * 4096 / 256, betas=(0.9, 0.95), weight_decay=0.04)
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
train_cfg = dict(max_epochs=800)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)

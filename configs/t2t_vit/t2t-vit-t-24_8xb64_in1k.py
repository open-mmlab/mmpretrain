_base_ = [
    '../_base_/models/t2t-vit-t-24.py',
    '../_base_/datasets/imagenet_bs64_t2t_224.py',
    '../_base_/default_runtime.py',
]

# optimizer
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={'cls_token': dict(decay_mult=0.0)},
)
optimizer = dict(
    type='AdamW',
    lr=5e-4,
    weight_decay=0.065,
    paramwise_cfg=paramwise_cfg,
)
optimizer_config = dict(grad_clip=None)

# learning policy
# FIXME: lr in the first 300 epochs conforms to the CosineAnnealing and
# the lr in the last 10 epoch equals to min_lr
lr_config = dict(
    policy='CosineAnnealingCooldown',
    min_lr=1e-5,
    cool_down_time=10,
    cool_down_ratio=0.1,
    by_epoch=True,
    warmup_by_epoch=True,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-6)
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]
runner = dict(type='EpochBasedRunner', max_epochs=310)

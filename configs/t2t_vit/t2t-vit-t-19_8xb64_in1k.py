_base_ = [
    '../_base_/models/t2t-vit-t-19.py',
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
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=10),
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        eta_min=1e-5,
        by_epoch=True,
        begin=10,
        end=300),
    dict(type='ConstantLR', factor=0.1, by_epoch=True, begin=300, end=310),
]
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=310)
val_cfg = dict(interval=1)  # validate every epoch
test_cfg = dict()

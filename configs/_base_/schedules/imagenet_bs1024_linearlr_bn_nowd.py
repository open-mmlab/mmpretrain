# optimizer
optimizer = dict(
    type='SGD',
    lr=0.5,
    momentum=0.9,
    weight_decay=0.00004,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
    warmup='constant',
    warmup_iters=5000,
)
runner = dict(type='EpochBasedRunner', max_epochs=300)

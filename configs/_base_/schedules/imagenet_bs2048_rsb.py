# optimizer
optimizer = dict(type='Lamb', lr=0.005, weight_decay=0.02)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1.0e-6,
    warmup='linear',
    warmup_iters=3150,
    warmup_ratio=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=100)

# Scheduler settings
optimizer = dict(
    type='AdamW', lr=5e-4, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=10,  # Warmup 10 epochs
    warmup_by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=300)

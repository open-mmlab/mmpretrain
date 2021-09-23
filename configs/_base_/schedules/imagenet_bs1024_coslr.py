# optimizer
optimizer = dict(type='SGD', lr=0.8, momentum=0.9, weight_decay=5e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)

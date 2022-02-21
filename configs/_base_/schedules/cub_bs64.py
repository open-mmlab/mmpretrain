# optimizer
optimizer = dict(
    type='SGD', lr=0.004, momentum=0.9, nesterov=True, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=50)

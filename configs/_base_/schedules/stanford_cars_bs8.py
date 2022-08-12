# optimizer
optimizer = dict(
    type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[40, 70, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)

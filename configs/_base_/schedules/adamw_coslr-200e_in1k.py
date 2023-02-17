# optimizer_wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05))

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
        type='CosineAnnealingLR', T_max=160, by_epoch=True, begin=40, end=200)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)

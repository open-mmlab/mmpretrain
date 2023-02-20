_base_ = 'beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k.py'

# drop_path_rate: 0. for 300 epochs and 0.1 for 1600 epochs.
model = dict(
    backbone=dict(drop_path_rate=0.1),
    neck=dict(drop_path_rate=0.1),
)

# optimizer wrapper
# betas: (0.9, 0.98) for 300 epochs and (0.9, 0.999) for 1600 epochs.
optimizer = dict(
    type='AdamW', lr=1.5e-3, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper', loss_scale='dynamic', optimizer=optimizer)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        begin=10,
        end=1600,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1600)

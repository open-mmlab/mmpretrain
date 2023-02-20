_base_ = 'mae_vit-base-p16_8xb512-amp-coslr-400e_in1k.py'

# pre-train for 800 epochs
train_cfg = dict(max_epochs=800)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.000000001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=760,
        by_epoch=True,
        begin=40,
        end=800,
        convert_to_iter_based=True)
]

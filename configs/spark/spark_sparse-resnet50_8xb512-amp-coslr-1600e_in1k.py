_base_ = 'spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k.py'

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
        type='CosineAnnealingLR',
        T_max=1560,
        by_epoch=True,
        begin=40,
        end=1600,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingWeightDecay',
        eta_min=0.2,
        T_max=1600,
        by_epoch=True,
        begin=0,
        end=1600,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(max_epochs=1600)

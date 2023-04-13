_base_ = 'spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k.py'

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=180,
        by_epoch=True,
        begin=20,
        end=200,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingWeightDecay',
        eta_min=0.2,
        T_max=200,
        by_epoch=True,
        begin=0,
        end=200,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(max_epochs=200)
resume = True

_base_ = 'barlowtwins_resnet50_8xb256-coslr-300e_in1k.py'

train_dataloader = dict(batch_size=32)

# optimizer
optim_wrapper = dict(accumulative_counts=8)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.6e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=90,
        eta_min=0.0016,
        by_epoch=True,
        begin=10,
        end=100,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)

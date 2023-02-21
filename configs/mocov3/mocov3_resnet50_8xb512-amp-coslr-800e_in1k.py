_base_ = 'mocov3_resnet50_8xb512-amp-coslr-100e_in1k.py'

model = dict(base_momentum=0.996)  # 0.99 for 100e and 300e, 0.996 for 800e

# optimizer
optim_wrapper = dict(optimizer=dict(lr=4.8, weight_decay=1.5e-6))

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
        T_max=790,
        by_epoch=True,
        begin=10,
        end=800,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800)

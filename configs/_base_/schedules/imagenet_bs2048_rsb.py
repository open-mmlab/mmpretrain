# optimizer
optimizer = dict(type='Lamb', lr=0.005, weight_decay=0.02)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=False,
        begin=0,
        end=5 * 626),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=100)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100)
val_cfg = dict(interval=1)  # validate every epoch
test_cfg = dict()

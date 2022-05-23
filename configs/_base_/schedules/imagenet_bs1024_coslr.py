# optimizer
optimizer = dict(type='SGD', lr=0.8, momentum=0.9, weight_decay=5e-5)
# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=95, by_epoch=True, begin=5, end=100)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300)
val_cfg = dict(interval=1)  # validate every epoch
test_cfg = dict()

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.25, by_epoch=False, begin=0,
        end=25025),
    dict(type='CosineAnnealingLR', T_max=195, by_epoch=True, begin=5, end=200)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200)
val_cfg = dict(interval=1)  # validate every epoch
test_cfg = dict()

# optimizer
optimizer = dict(
    type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=False,
        begin=0,
        end=5 * 626),
    dict(type='CosineAnnealingLR', T_max=95, by_epoch=True, begin=5, end=100)
]
# old learning policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_iters=2500,
#     warmup_ratio=0.25)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100)
val_cfg = dict(interval=1)  # validate every epoch
test_cfg = dict()

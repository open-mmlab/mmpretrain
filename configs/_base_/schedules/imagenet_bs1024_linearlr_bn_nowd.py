# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(norm_decay_mult=0),
)

# learning policy
param_scheduler = [
    dict(type='ConstantLR', factor=0.1, by_epoch=False, begin=0, end=5 * 1252),
    dict(type='PolyLR', eta_min=0, by_epoch=True, begin=5, end=300)
]

# old learning policy
# lr_config = dict(
#     policy='poly',
#     min_lr=0,
#     by_epoch=False,
#     warmup='constant',
#     warmup_iters=5000,
# )

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300)
val_cfg = dict(interval=1)  # validate every other epoch
test_cfg = dict()

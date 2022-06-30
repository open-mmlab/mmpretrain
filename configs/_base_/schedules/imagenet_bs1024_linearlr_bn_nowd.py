# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(norm_decay_mult=0),
)

# learning policy
param_scheduler = [
    dict(
        type='ConstantLR',
        factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(type='PolyLR', eta_min=0, by_epoch=True, begin=5, end=300)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300)
val_cfg = dict(interval=1)  # validate every other epoch
test_cfg = dict()

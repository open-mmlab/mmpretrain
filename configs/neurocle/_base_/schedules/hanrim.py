# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.0001, weight_decay=0.0001))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        by_epoch=True,
        begin=5,
        end=50,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)

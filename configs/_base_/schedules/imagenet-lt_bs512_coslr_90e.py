train_epochs = 90
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0005))

# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR',
    eta_min=0,
    T_max=train_epochs,
    by_epoch=True,  # Notice, by_epoch need to be set to True
    convert_to_iter_based=True  # convert to an iter-based scheduler)
)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=train_epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=512)

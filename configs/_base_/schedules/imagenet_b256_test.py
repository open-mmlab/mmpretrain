# optimizer
N = 60
warmup_epochs = 10
optim_wrapper = dict(
    optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.3),)


# # learning policy
# param_scheduler = dict(
#     type='MultiStepLR', by_epoch=True, milestones=[N-20, N-10], gamma=0.1)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='MultiStepLR', by_epoch=True, milestones=[N - 20, N - 10], gamma=0.1)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=N, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)

_base_ = [
    '../_base_/models/mobileone/mobileone_s0.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/default_runtime.py'
]

# dataset settings
train_dataloader = dict(batch_size=128)
val_dataloader = dict(batch_size=128)
test_dataloader = dict(batch_size=128)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=300),
    dict(
        type='CosineAnnealingParamScheduler',
        param_name='weight_decay',
        eta_min=0.00001,
        by_epoch=True,
        begin=0,
        end=300)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1024)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=5e-4, priority='ABOVE_NORMAL')]

_base_ = [
    '../_base_/models/convnext_v2/nano.py',
    '../_base_/datasets/imagenet_bs64_swin_384.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# dataset setting
train_dataloader = dict(batch_size=32)

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=8e-4, weight_decay=0.3),
    clip_grad=None,
)

# learning policy
param_scheduler = [dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True)]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

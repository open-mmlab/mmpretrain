_base_ = [
    '../_base_/models/convnext/convnext-large.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# dataset setting
train_dataloader = dict(batch_size=64)

# schedule setting
optimizer = dict(lr=4e-3)

# runtime setting
default_hooks = dict(optimizer=dict(grad_clip=dict(max_norm=5.0)))
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

_base_ = [
    '../_base_/models/edgenext/edgenext-small.py',
    '../_base_/datasets/imagenet_bs64_edgenext_256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=6e-3),
    clip_grad=dict(max_norm=5.0),
)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=4096)

_base_ = [
    '../_base_/models/hornet/hornet-large-gf384.py',
    '../_base_/datasets/imagenet_bs64_swin_384.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

data = dict(samples_per_gpu=16)

optimizer = dict(lr=5e-5)
optimizer_config = dict(grad_clip=dict(max_norm=1.0), _delete_=True)

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

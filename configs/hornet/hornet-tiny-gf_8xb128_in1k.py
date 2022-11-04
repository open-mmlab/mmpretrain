_base_ = [
    '../_base_/models/hornet/hornet-tiny-gf.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

data = dict(samples_per_gpu=128)

optim_wrapper = dict(optimizer=dict(lr=4e-3), clip_grad=dict(max_norm=1.0))

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

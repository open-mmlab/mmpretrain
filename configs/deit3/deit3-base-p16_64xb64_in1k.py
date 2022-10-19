_base_ = [
    '../_base_/models/deit3/deit3-base-p16-224.py',
    '../_base_/datasets/imagenet_bs256_deit3.py',
    '../_base_/schedules/imagenet_bs2048_deit3.py',
    '../_base_/default_runtime.py'
]

# runtime settings
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

_base_ = [
    '../_base_/models/edgenext/edgenext-xsmall.py',
    '../_base_/datasets/imagenet_bs64_edgenext_256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

data = dict(samples_per_gpu=256)

optimizer = dict(lr=6e-3)

_base_ = [
    '../_base_/models/eva/eva-g.py',
    '../_base_/datasets/imagenet_bs16_eva_560.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(img_size=560))

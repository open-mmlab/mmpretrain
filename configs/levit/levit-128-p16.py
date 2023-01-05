_base_ = [
    '../_base_/models/levit-256-p16.py',
    '../_base_/datasets/imagenet_bs256_levit_224.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/imagenet_bs1024_adamw_levit.py'
]

model = dict(backbone=dict(arch='128'), head=dict(in_channels=384, ))

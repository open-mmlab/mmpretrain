_base_ = [
    '../_base_/models/levit-256-p16.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs2048_adamw_levit.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(backbone=dict(arch='192'), head=dict(in_channels=384))

# dataset settings
train_dataloader = dict(batch_size=256)

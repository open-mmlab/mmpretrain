_base_ = [
    '../_base_/models/convmixer/convmixer-1536-20.py',
    '../_base_/datasets/imagenet_bs64_convmixer_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

default_hooks = dict(optimizer=dict(grad_clip=dict(max_norm=5.0)))

optimizer = dict(lr=0.01)

train_cfg = dict(by_epoch=True, max_epochs=150)

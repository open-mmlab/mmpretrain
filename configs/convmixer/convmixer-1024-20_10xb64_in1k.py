_base_ = [
    '../_base_/models/convmixer/convmixer-1024-20.py',
    '../_base_/datasets/imagenet_bs64_convmixer_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

optimizer = dict(lr=0.01)

runner = dict(type='EpochBasedRunner', max_epochs=150)

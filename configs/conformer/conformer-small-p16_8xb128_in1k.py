_base_ = [
    '../_base_/models/conformer/small-p16.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu=128)
evaluation = dict(interval=1, metric='accuracy')

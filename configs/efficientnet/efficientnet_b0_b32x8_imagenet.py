_base_ = [
    '../_base_/models/efficientnet_b0.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

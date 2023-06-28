_base_ = [
    '../_base_/models/efficientformer-l1_custom.py',
    '../_base_/datasets/imagenet_bs128_poolformer_small_224_custom_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

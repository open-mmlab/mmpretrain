# Only for evaluation
_base_ = [
    '../_base_/models/swin_transformer/large_384.py',
    '../_base_/datasets/imagenet_bs64_swin_384.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

_base_ = [
    '../_base_/models/swin_transformer/base_384.py',
    '../_base_/datasets/imagenet_bs128_swin_384.py',
    '../_base_/schedules/imagenet_bs2048_AdamW.py',
    '../_base_/default_runtime.py'
]

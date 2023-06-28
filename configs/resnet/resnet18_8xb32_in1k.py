_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/datasets/imagenet_bs32.py',
    # '../_base_/schedules/imagenet_b256_test.py',
    '../_base_/schedules/imagenet_bs2048_AdamW.py',
    '../_base_/default_runtime.py'
]

_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs2048_coslr.py',
    '../_base_/default_runtime.py'
]

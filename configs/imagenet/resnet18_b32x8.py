_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32_lmdb.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

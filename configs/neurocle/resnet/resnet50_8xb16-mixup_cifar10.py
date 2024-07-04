_base_ = [
    '../_base_/models/resnet50_cifar_mixup.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

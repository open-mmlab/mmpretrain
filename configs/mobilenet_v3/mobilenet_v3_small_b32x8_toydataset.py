_base_ = [
    '../_base_/models/mobilenet_v3_small.py',
    # '../_base_/datasets/toydataset.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py'
    # '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]

# runner = dict(type='EpochBasedRunner', max_epochs=30)
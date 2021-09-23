_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs1024_coslr.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(head=dict(num_classes=21843))

# dataset settings
dataset_type = 'ImageNet21k'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet22k/train',
        recursion_subdir=True),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet22k/val',
        ann_file='data/imagenet22k/meta/val.txt',
        recursion_subdir=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/imagenet22k/val',
        ann_file='data/imagenet22k/meta/val.txt',
        recursion_subdir=True))

runner = dict(type='EpochBasedRunner', max_epochs=90)

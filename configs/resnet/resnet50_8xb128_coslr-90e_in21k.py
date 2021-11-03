_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet21k_bs128.py',
    '../_base_/schedules/imagenet_bs1024_coslr.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(head=dict(num_classes=21843))

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=90)

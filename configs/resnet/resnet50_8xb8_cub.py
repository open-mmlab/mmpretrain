_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cub_bs8_448.py',
    '../_base_/schedules/cub_bs64.py',
    '../_base_/default_runtime.py',
]

# model settings
# use pre-train weight converted from https://github.com/Alibaba-MIIL/ImageNet21K # noqa
pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'  # noqa

model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained, prefix='backbone')),
    head=dict(num_classes=200, ))

# runtime settings
default_hooks = dict(logger=dict(type='LoggerHook', interval=20))

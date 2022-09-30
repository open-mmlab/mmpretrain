_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/stanford_cars_bs8_448.py',
    '../_base_/schedules/stanford_cars_bs8.py', '../_base_/default_runtime.py'
]

# use pre-train weight converted from https://github.com/Alibaba-MIIL/ImageNet21K # noqa
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'  # noqa

model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    head=dict(num_classes=196, ))

log_config = dict(interval=50)
checkpoint_config = dict(
    interval=1, max_keep_ckpts=3)  # save last three checkpoints

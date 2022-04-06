_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/cub_bs8_448.py',
    '../_base_/schedules/cub_bs64.py', '../_base_/default_runtime.py'
]

# use pre-train weight converted from https://github.com/Alibaba-MIIL/ImageNet21K # noqa
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'  # noqa

model = dict(
    type='ImageClassifier',
    backbone=[
        dict(type='LearnableResizer', output_size=(384, 384)),
        dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint=checkpoint, prefix='backbone'))
    ],
    head=dict(num_classes=200, ))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=1200),
    dict(type='RandomCrop', size=896),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=1200),
    dict(type='CenterCrop', crop_size=896),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

optimizer = dict(lr=0.01 / 8)

log_config = dict(interval=20)  # log every 20 intervals

checkpoint_config = dict(
    interval=1, max_keep_ckpts=3)  # save last three checkpoints

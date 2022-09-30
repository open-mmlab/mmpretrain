_base_ = ['../_base_/datasets/voc_bs16.py', '../_base_/default_runtime.py']

# Pre-trained Checkpoint Path
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth'  # noqa
# If you want to use the pre-trained weight of ResNet101-CutMix from
# the originary repo(https://github.com/Kevinz-code/CSRA). Script of
# 'tools/convert_models/torchvision_to_mmcls.py' can help you convert weight
# into mmcls format. The mAP result would hit 95.5 by using the weight.
# checkpoint = 'PATH/TO/PRE-TRAINED_WEIGHT'

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    neck=None,
    head=dict(
        type='CSRAClsHead',
        num_classes=20,
        in_channels=2048,
        num_heads=1,
        lam=0.1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

# dataset setting
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=448, scale=(0.7, 1.0)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=448),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    # map the difficult examples as negative ones(0)
    train=dict(pipeline=train_pipeline, difficult_as_postive=False),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
# the lr of classifier.head is 10 * base_lr, which help convergence.
optimizer = dict(
    type='SGD',
    lr=0.0002,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10)}))

optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    step=6,
    gamma=0.1,
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1e-7,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=20)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=3,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)

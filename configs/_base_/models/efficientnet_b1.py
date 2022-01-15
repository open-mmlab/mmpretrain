# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNet',
        arch='b1',
        conv_cfg=dict(type='Conv2dAdaptivePadding'),
        norm_cfg=dict(type='BN', eps=1e-3)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

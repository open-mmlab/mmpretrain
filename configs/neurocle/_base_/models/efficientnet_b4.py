# model settings
model = dict(
    type='ImageClassifier',
    pretrained='https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32_in1k_20220119-81fd4077.pth',
    backbone=dict(type='EfficientNet', arch='b4'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1792,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

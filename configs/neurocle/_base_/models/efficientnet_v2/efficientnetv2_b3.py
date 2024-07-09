# model settings
model = dict(
    type='ImageClassifier',
    pretrained='https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-b3_3rdparty_in1k_20221221-b6f07a36.pth',
    backbone=dict(type='EfficientNetV2', arch='b3'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

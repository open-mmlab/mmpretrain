# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='RegNet', arch='regnety_6.4gf'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1296,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

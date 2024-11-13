# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='StarNet', arch='s1'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=9,
        in_channels=192,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

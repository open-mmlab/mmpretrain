# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',  #'AngularPenaltyHead',
        num_classes=3731,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),  #
    ))

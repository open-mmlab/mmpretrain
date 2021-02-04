# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetv3', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ConvClsHead',
        num_classes=2,
        in_channels=576,
        mid_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1)
    ))
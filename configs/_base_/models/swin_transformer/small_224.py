# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='SwinTransformer', arch='small', img_size=224),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

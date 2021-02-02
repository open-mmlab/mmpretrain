# model settings
model = dict(
    type='ImageClassifier',
    # TODO: arch
    backbone=dict(type='MobileNetv3', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ConvClsHead',
        # TODO: class
        num_classes=1000,
        # num_classes=2,
        in_channels=576,
        mid_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        # TODO: 
        topk=(1, 5),
        # topk=(1),
    ))

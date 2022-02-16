# model settings
dict(
    type='ImageClassifier',
    backbone=dict(
        type='RepMLPNet',
        arch='B224',
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        reparam_conv_k=(1, 3),
        deploy=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

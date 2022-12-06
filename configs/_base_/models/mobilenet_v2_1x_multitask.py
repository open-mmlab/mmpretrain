# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/mobi\
            lenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
            prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiTaskHead',
        task_heads={
            'damage_severity': dict(type='LinearClsHead', num_classes=3),
            'informative': dict(type='LinearClsHead', num_classes=2),
            'humanitarian': dict(type='LinearClsHead', num_classes=4),
            'disaster_types': dict(type='LinearClsHead', num_classes=7)
        },
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

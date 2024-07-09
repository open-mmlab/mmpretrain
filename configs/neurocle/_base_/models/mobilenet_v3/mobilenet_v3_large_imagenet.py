# model settings
model = dict(
    type='ImageClassifier',
    pretrained='https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-large_8xb128_in1k_20221114-0ed9ed9a.pth', 
    backbone=dict(type='MobileNetV3', arch='large'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=1000,
        in_channels=960,
        mid_channels=[1280],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(
            type='Normal', layer='Linear', mean=0., std=0.01, bias=0.),
        topk=(1, 5)))

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MLPMixer',
        in_channels=3,
        image_size=224,
        patch_size=16,
        dim=512,
        num_layers=8,
        token_dim=256,
        channel_dim=2048
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1),
    ))
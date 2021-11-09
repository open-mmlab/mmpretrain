# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MlpMixer',
        patch_size=16,
        num_blocks=24,
        embed_dims=1024,
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

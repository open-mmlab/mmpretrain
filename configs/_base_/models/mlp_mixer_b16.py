# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MlpMixer',
        patch_size=16,
        num_blocks=12,
        embed_dims=768,
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

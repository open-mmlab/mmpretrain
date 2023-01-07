# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='XCiT',
        patch_size=16,
        embed_dim=128,
        depth=12,
        num_heads=4,
        eta=1.0,
        tokens_norm=False,
        img_size=384,
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=128,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

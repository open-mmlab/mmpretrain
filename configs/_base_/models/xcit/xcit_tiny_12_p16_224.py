# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='XCiT',
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=4,
        eta=1.0,
        tokens_norm=True,
        img_size=224,
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=192,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

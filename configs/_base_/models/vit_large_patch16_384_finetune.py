# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        num_layers=24,
        embed_dim=1024,
        num_heads=16,
        img_size=384,
        patch_size=16,
        in_channels=3,
        feedforward_channels=4096,
        drop_rate=0.1),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

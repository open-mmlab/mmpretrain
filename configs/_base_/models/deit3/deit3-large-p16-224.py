model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeiT3',
        arch='l',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.45),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='BinaryCrossEntropyLoss', target_threshold=0.0),
        init_cfg=None,
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

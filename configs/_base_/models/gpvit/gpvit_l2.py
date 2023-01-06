model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='GPViT',
        arch='L2',
        img_size=224,
        drop_path_rate=0.2,
    ),
    neck=dict(type='GroupNeck', embed_dims=348),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=348,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1, 5)),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

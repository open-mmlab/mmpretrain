# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RevVisionTransformer',
        arch='deit-base',
        img_size=224,
        patch_size=16,
        output_cls_token=False,
        avg_token=True,
        with_cls_token=False,
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)

# model settings
embed_dims = 448
num_classes = 1000

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='T2T_ViT',
        img_size=224,
        in_channels=3,
        embed_dims=embed_dims,
        t2t_cfg=dict(
            token_dims=64,
            use_performer=False,
        ),
        num_layers=19,
        layer_cfgs=dict(
            num_heads=7,
            feedforward_channels=3 * embed_dims,  # mlp_ratio = 3
        ),
        drop_path_rate=0.1,
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=.02),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=num_classes,
        in_channels=embed_dims,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
        ),
        topk=(1, 5),
        init_cfg=dict(type='TruncNormal', layer='Linear', std=.02)),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, prob=0.5, num_classes=num_classes),
        dict(type='BatchCutMix', alpha=1.0, prob=0.5, num_classes=num_classes),
    ]))

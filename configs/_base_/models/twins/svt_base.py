# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SVT',
        in_channels=3,
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        patch_sizes=[4, 2, 2, 2],
        strides=[4, 2, 2, 2],
        mlp_ratios=[4, 4, 4, 4],
        out_indices=(3, ),
        qkv_bias=True,
        norm_cfg=dict(type='LN'),
        depths=[2, 2, 18, 2],
        sr_ratios=[8, 4, 2, 1],
        windiow_sizes=[7, 7, 7, 7],
        norm_after_stage=False,
        drop_rate=0.0,
        attn_drop_rate=0.,
        drop_path_rate=0.3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))

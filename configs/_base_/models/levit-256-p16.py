# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='LeViT',
        img_size=224,
        patch_size=16,
        drop_path=0,
        embed_dim=[256, 384, 512],
        num_heads=[4, 6, 8],
        depth=[4, 4, 4],
        key_dim=[32, 32, 32],
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', 32, 256 // 32, 4, 2, 2],
            ['Subsample', 32, 384 // 32, 4, 2, 2],
        ],
        out_indices=(2, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LeViTClsHead',
        num_classes=1000,
        in_channels=512,
        distillation=False,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1, 5),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]))

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
        key_dim=[32] * 3,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', 32, 8, 4, 2, 2],
            ['Subsample', 32, 12, 4, 2, 2],
        ],
        out_indices=(2,)

    ),
    neck=None,
    head=dict(
        type='LeViTClsHead',
        num_classes=1000,
        in_channels=512,
        distillation=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

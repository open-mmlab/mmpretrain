# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        embed_dim=1024,
        img_size=384,
        patch_size=16,
        in_channels=3,
        drop_rate=0.1,
        hybrid_backbone=None,
        encoder=dict(
            type='VitTransformerEncoder',
            num_layers=24,
            transformerlayers=dict(
                type='VitTransformerEncoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=1024,
                        num_heads=16,
                        attn_drop=0.,
                        proj_drop=0.1)
                ],
                feedforward_channels=4096,
                ffn_dropout=0.1,
                operation_order=('norm', 'self_attn', 'norm', 'ffn'))),
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

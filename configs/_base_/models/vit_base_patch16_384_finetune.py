# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        embed_dim=768,
        img_size=384,
        patch_size=16,
        in_channels=3,
        drop_rate=0.1,
        hybrid_backbone=None,
        encoder=dict(
            type='TransformerEncoder',
            num_layers=12,
            transformerlayers=dict(
                type='VitTransformerEncoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=768,
                        num_heads=12,
                        # FIXME dropout should be replaced with
                        #  attn_drop and proj_drop
                        dropout=0.1)
                ],
                feedforward_channels=3072,
                ffn_dropout=0.1,
                operation_order=('norm', 'self_attn', 'norm', 'ffn')),
            init_cfg=[
                dict(type='Xavier', layer='Linear', distribution='normal')
            ]),
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
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

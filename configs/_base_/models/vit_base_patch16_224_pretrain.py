# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        embed_dim=768,
        img_size=224,
        patch_size=16,
        in_channels=3,
        drop_rate=0.1,
        hybrid_backbone=None,
        encoder=dict(
            type='VitTransformerEncoder',
            num_layers=12,
            transformerlayers=dict(
                type='VitTransformerEncoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=768,
                        num_heads=12,
                        attn_drop=0.,
                        dropout_layer=dict(type='DropOut', drop_prob=0.1))
                ],
                ffn_cfgs=dict(
                    embed_dims=768,
                    feedforward_channels=3072,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='GELU')),
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
        in_channels=768,
        hidden_dim=3072,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
        topk=(1, 5),
    ),
    train_cfg=dict(mixup=dict(alpha=0.2, num_classes=1000)))

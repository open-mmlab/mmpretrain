# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        in_channels=3,
        drop_rate=0.1,
        attn_drop_rate=0.,
        hybrid_backbone=None,
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
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=1000,
                      prob=1.)))

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='l',
        img_size=336,
        patch_size=14,
        drop_rate=0.1,
        avg_token=True,
        output_cls_token=False,
        pre_norm=True,
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ))

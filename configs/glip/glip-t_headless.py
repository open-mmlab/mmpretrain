model = dict(
    type='ImageClassifier',
    backbone=dict(type='SwinTransformer', arch='tiny', img_size=224),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    # convert image from BGR to RGB
    to_rgb=False,
)

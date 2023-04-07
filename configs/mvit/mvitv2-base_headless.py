model = dict(
    type='ImageClassifier',
    backbone=dict(type='MViT', arch='base', drop_path_rate=0.3),
    neck=None,
    head=None)

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

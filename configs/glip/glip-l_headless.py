model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='large',
        img_size=384,
        out_indices=(1, 2, 3),  # original weight is for detection
        stage_cfgs=dict(block_cfgs=dict(window_size=12))),
    neck=None,
    head=None)

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    # convert image from BGR to RGB
    to_rgb=False,
)

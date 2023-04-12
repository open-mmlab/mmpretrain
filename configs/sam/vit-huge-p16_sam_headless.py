# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ViTSAM',
        arch='huge',
        img_size=1024,
        patch_size=16,
        out_channels=256,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
    ),
    neck=None,
    head=None,
)

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

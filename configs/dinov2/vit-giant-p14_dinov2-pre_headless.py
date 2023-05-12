# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='dinov2-giant',
        img_size=518,
        patch_size=14,
        layer_scale_init_value=1e-5,
        layer_cfgs=dict(ffn_type='swiglu_fused'),
    ),
    neck=None,
    head=None)

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

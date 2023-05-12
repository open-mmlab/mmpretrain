model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ViTEVA02',
        arch='l',
        img_size=224,
        patch_size=14,
        sub_ln=True,
        final_norm=False,
        out_type='avg_featmap'),
    neck=None,
    head=None,
)

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    # convert image from BGR to RGB
    to_rgb=True,
)

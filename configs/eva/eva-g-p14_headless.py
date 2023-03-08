model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BEiTViT',
        arch='eva-g',
        img_size=224,
        patch_size=14,
        layer_scale_init_value=0.0,
        out_type='avg_featmap',
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
    ),
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

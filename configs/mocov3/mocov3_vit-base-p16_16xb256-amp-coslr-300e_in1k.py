_base_ = 'mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k.py'

# model settings
model = dict(
    backbone=dict(arch='base'),  # embed_dim = 768
    neck=dict(in_channels=768),
)

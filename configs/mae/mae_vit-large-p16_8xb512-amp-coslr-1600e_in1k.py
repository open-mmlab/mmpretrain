_base_ = 'mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k.py'

# model settings
model = dict(
    backbone=dict(type='MAEViT', arch='l', patch_size=16, mask_ratio=0.75),
    neck=dict(type='MAEPretrainDecoder', embed_dim=1024))

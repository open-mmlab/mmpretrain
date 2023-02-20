_base_ = 'mae_vit-large-p16_8xb512-amp-coslr-300e_in1k.py'

# pre-train for 100 epochs
train_cfg = dict(max_epochs=1600)

# model settings
model = dict(
    backbone=dict(type='MAEViT', arch='h', patch_size=14, mask_ratio=0.75),
    neck=dict(
        type='MAEPretrainDecoder',
        embed_dim=1280,
        patch_size=14,
        num_patches=256),
    head=dict(patch_size=14))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=1560,
        by_epoch=True,
        begin=40,
        end=1600,
        convert_to_iter_based=True)
]

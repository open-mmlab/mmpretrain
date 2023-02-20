_base_ = [
    '../_base_/datasets/imagenet_bs256_maskfeat.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MaskFeat',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(type='MaskFeatViT', arch='b', patch_size=16),
    neck=dict(
        type='LinearNeck',
        in_channels=768,
        out_channels=108,
        with_avg_pool=False,
        init_cfg=dict(type='TruncNormal', layer='Linear', std=0.02, bias=0)),
    head=dict(
        type='MaskFeatPretrainHead',
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    target_generator=dict(
        type='HOGGenerator', nbins=9, pool=8, gaussian_window=16))

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=2e-4 * 8, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=0.02),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        # commented 'pos_embed' and 'cls_token' to avoid loss stuck situation
        custom_keys={
            # 'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            # 'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=30,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=270,
        by_epoch=True,
        begin=30,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
# pre-train for 300 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

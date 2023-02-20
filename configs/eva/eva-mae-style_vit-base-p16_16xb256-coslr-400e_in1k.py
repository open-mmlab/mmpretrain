_base_ = [
    '../_base_/datasets/imagenet_bs128_mae.py',
    '../_base_/default_runtime.py',
]

# dataset settings
train_dataloader = dict(batch_size=256, num_workers=8)

# model settings
model = dict(
    type='EVA',
    backbone=dict(
        type='MAEViT',
        arch='b',
        patch_size=16,
        mask_ratio=0.75,
        init_cfg=[
            dict(type='Xavier', distribution='uniform', layer='Linear'),
            dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    neck=dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        predict_feature_dim=512,
        init_cfg=[
            dict(type='Xavier', distribution='uniform', layer='Linear'),
            dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    head=dict(
        type='MILANPretrainHead',
        loss=dict(
            type='CosineSimilarityLoss', shift_factor=2.0, scale_factor=2.0),
    ),
    target_generator=dict(
        type='CLIPGenerator',
        tokenizer_path=  # noqa
        'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/clip_vit_base_16.pth.tar'  # noqa
    ),
    init_cfg=None)

# optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))
find_unused_parameters = True

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
        T_max=360,
        by_epoch=True,
        begin=40,
        end=400,
        convert_to_iter_based=True)
]

# runtime settings
# pre-train for 400 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)
resume = True

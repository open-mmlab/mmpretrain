_base_ = [
    '../_base_/datasets/imagenet_bs256_itpn.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='iTPN',
    backbone=dict(
        type='iTPNHiViT',
        arch='base',
        drop_path_rate=0.1,
        rpe=True,
        layer_scale_init_value=0.1,
        reconstruction_type='clip'),
    neck=dict(
        type='iTPNPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        mlp_ratio=4.,
        reconstruction_type='clip',
        #  transformer pyramid
        fpn_dim=256,
        fpn_depth=2,
        num_outs=3,
    ),
    head=dict(
        type='iTPNClipHead',
        embed_dims=512,
        num_embed=512,
        loss=dict(type='CrossEntropyLoss')),
    target_generator=dict(
        type='CLIPGenerator',
        tokenizer_path=  # noqa
        'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/clip_vit_base_16.pth.tar'  # noqa
    ),
)

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    # betas: (0.9, 0.98) for 300 epochs and (0.9, 0.999) for 800/1600 epochs.
    optimizer=dict(
        type='AdamW', lr=1.5e-3, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=3.0),
    paramwise_cfg=dict(
        custom_keys={
            '.norm': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.gamma': dict(decay_mult=0.0),
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        begin=10,
        end=800,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

randomness = dict(seed=0, diff_rank_seed=True)

find_unused_parameters = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)

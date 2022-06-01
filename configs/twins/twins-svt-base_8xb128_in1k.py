_base_ = [
    '../_base_/models/twins_svt_base.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# dataset settings
train_dataloader = dict(batch_size=128)

# schedule settings
paramwise_cfg = dict(_delete=True, norm_decay_mult=0.0, bias_decay_mult=0.0)

optimizer = dict(
    type='AdamW',
    lr=5e-4 * 128 * 8 / 512,  # learning rate for 128 batch size, 8 gpu.
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)

param_scheduler = [
    # warm up learning rate schedule
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1e-2,
        by_epoch=True,
        begin=5,
        end=300)
]

# runtime settings
default_hooks = dict(optimizer=dict(grad_clip=dict(max_norm=5.0)))

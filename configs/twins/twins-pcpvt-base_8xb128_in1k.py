_base_ = [
    '../_base_/models/twins_pcpvt_base.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

default_hooks = dict(
    optimizer=dict(_delete_=True, grad_clip=dict(max_norm=5.0)))

data = dict(samples_per_gpu=128)

paramwise_cfg = dict(_delete=True, norm_decay_mult=0.0, bias_decay_mult=0.0)

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 128 * 8 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1e-2,
        by_epoch=True,
        begin=5,
        end=300)
]

evaluation = dict(interval=1, metric='accuracy')

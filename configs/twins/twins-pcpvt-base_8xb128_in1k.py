_base_ = [
    '../_base_/models/twins_pcpvt_base.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

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
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=True,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=5,
    warmup_by_epoch=True)

evaluation = dict(interval=1, metric='accuracy')

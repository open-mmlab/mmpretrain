_base_ = [
    '../_base_/models/mvit/mvitv2-tiny.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs2048_AdamW.py',
    '../_base_/default_runtime.py'
]

# dataset settings
data = dict(samples_per_gpu=256)

# schedule settings
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.pos_embed': dict(decay_mult=0.0),
        '.rel_pos_h': dict(decay_mult=0.0),
        '.rel_pos_w': dict(decay_mult=0.0)
    })

optimizer = dict(lr=0.00025, paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=70,
    warmup_by_epoch=True)

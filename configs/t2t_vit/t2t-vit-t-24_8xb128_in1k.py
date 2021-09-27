_base_ = [
    '../_base_/models/t2t-vit-t-24.py',
    '../_base_/datasets/imagenet_bs128_t2t_224.py',
    '../_base_/default_runtime.py',
]

# optimizer
paramwise_cfg = dict(
    bias_decay_mult=0.0,
    custom_keys={'.backbone.cls_token': dict(decay_mult=0.0)},
)
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    paramwise_cfg=paramwise_cfg,
)
optimizer_config = dict(grad_clip=None)

# learning policy
# FIXME: lr in the first 300 epochs conforms to the CosineAnnealing and
# the lr in the last 10 epoch equals to min_lr
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    by_epoch=True,
    warmup_by_epoch=True,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=310)

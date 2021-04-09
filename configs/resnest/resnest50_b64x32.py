_base_ = [
    '../_base_/models/resnest50_train.py',
    '../_base_/datasets/imagenet_bs64_ra.py', '../_base_/default_runtime.py'
]

# batch_size = 4096 8x8gpus
# optimizer
optimizer = dict(
    type='SGD',
    lr=1.6,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=270)

_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet_bs256_rsb_a3.py',
    '../_base_/schedules/imagenet_bs2048_rsb.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
    head=dict(loss=dict(use_sigmoid=True)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1),
        dict(type='CutMix', alpha=1.0)
    ]),
)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(lr=0.008),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

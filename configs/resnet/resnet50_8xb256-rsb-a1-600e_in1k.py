_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet_bs256_rsb_a12.py',
    '../_base_/schedules/imagenet_bs2048_rsb.py',
    '../_base_/default_runtime.py'
]

# Model settings
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        drop_path_rate=0.05,
    ),
    head=dict(
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
        )),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.2, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))

# Dataset settings
sampler = dict(type='RepeatAugSampler')

# Schedule settings
runner = dict(max_epochs=600)
optimizer = dict(
    weight_decay=0.01,
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

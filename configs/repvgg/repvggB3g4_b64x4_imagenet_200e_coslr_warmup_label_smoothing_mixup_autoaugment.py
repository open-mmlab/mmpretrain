_base_ = './repvggA0_b64x4_imagenet.py'

model = dict(
    backbone=dict(arch='B3g4'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2560,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            mode='classy_vision',
            num_classes=1000)),
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=1000,
                      prob=1.)))

sub_policy_0 = [
    dict(type='Posterize', bits=8, prob=0.4),
    dict(type='Rotate', angle=9., prob=0.6)
]
sub_policy_1 = [
    dict(type='Solarize', thr=5, prob=0.6),
    dict(type='AutoContrast', prob=0.6)
]
sub_policy_2 = [
    dict(type='Equalize', prob=0.8),
    dict(type='Equalize', prob=0.6)
]
sub_policy_3 = [
    dict(type='Posterize', bits=7, prob=0.6),
    dict(type='Posterize', bits=6, prob=0.6)
]
sub_policy_4 = [
    dict(type='Equalize', prob=0.4),
    dict(type='Solarize', thr=4, prob=0.2)
]
sub_policy_5 = [
    dict(type='Equalize', prob=0.4),
    dict(type='Rotate', angle=8., prob=0.8)
]
sub_policy_6 = [
    dict(type='Solarize', thr=3, prob=0.6),
    dict(type='Equalize', prob=0.6)
]
sub_policy_7 = [
    dict(type='Posterize', bits=5, prob=0.8),
    dict(type='Equalize', prob=1.0)
]
sub_policy_8 = [
    dict(type='Rotate', angle=3., prob=0.2),
    dict(type='Solarize', thr=8, prob=0.6)
]
sub_policy_9 = [
    dict(type='Equalize', prob=0.6),
    dict(type='Posterize', bits=6, prob=0.4)
]
sub_policy_10 = [
    dict(type='Rotate', angle=8., prob=0.8),
    dict(type='ColorTransform', magnitude=0, prob=0.4)
]
sub_policy_11 = [
    dict(type='Rotate', angle=9., prob=0.4),
    dict(type='Equalize', prob=0.6)
]
sub_policy_12 = [
    dict(type='Equalize', prob=0.0),
    dict(type='Equalize', prob=0.8)
]
sub_policy_13 = [
    dict(type='Invert', prob=0.6),
    dict(type='Equalize', prob=1.0)
]
sub_policy_14 = [
    dict(type='ColorTransform', magnitude=4, prob=0.6),
    dict(type='Contrast', magnitude=8, prob=1.0)
]
sub_policy_15 = [
    dict(type='Rotate', angle=8., prob=0.8),
    dict(type='ColorTransform', magnitude=2, prob=1.0)
]
sub_policy_16 = [
    dict(type='ColorTransform', magnitude=8, prob=0.8),
    dict(type='Solarize', thr=7, prob=0.8)
]
sub_policy_17 = [
    dict(type='Sharpness', magnitude=7, prob=0.4),
    dict(type='Invert', prob=0.6)
]
sub_policy_18 = [
    dict(type='Shear', magnitude=5, prob=0.6),
    dict(type='Equalize', prob=1.0)
]
sub_policy_19 = [
    dict(type='ColorTransform', magnitude=0, prob=0.4),
    dict(type='Equalize', prob=0.6)
]
sub_policy_20 = [
    dict(type='Equalize', prob=0.4),
    dict(type='Solarize', thr=4, prob=0.2)
]
sub_policy_21 = [
    dict(type='Solarize', thr=5, prob=0.6),
    dict(type='AutoContrast', prob=0.6)
]
sub_policy_22 = [
    dict(type='Invert', prob=0.6),
    dict(type='Equalize', prob=1.0)
]
sub_policy_23 = [
    dict(type='ColorTransform', magnitude=4, prob=0.6),
    dict(type='Contrast', magnitude=8, prob=1.0)
]

policies = [
    sub_policy_0, sub_policy_1, sub_policy_2, sub_policy_3, sub_policy_4,
    sub_policy_5, sub_policy_6, sub_policy_7, sub_policy_8, sub_policy_9,
    sub_policy_10, sub_policy_11, sub_policy_12, sub_policy_13, sub_policy_14,
    sub_policy_15, sub_policy_16, sub_policy_17, sub_policy_18, sub_policy_19,
    sub_policy_20, sub_policy_21, sub_policy_22, sub_policy_23
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='AutoAugment', policies=policies),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
data = dict(train=dict(pipeline=train_pipeline))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=25025,
    warmup_ratio=0.25)
runner = dict(max_epochs=200)

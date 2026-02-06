checkpoint_config = dict(interval=1)
data = dict(
    samples_per_gpu=256,
    test=dict(
        _scope_='mmcls',
        ann_file='data/imagenet/meta/val.txt',
        data_prefix='data/imagenet/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(size=(
                236,
                -1,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='ImageNet'),
    train=dict(
        _scope_='mmcls',
        data_prefix='data/imagenet/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(size=224, type='RandomResizedCrop'),
            dict(direction='horizontal', flip_prob=0.5, type='RandomFlip'),
            dict(
                hparams=dict(
                    interpolation='bicubic', pad_val=[
                        104,
                        116,
                        124,
                    ]),
                magnitude_level=7,
                magnitude_std=0.5,
                num_policies=2,
                policies=[
                    dict(_scope_='mmcls', type='AutoContrast'),
                    dict(_scope_='mmcls', type='Equalize'),
                    dict(_scope_='mmcls', type='Invert'),
                    dict(
                        _scope_='mmcls',
                        magnitude_key='angle',
                        magnitude_range=(
                            0,
                            30,
                        ),
                        type='Rotate'),
                    dict(
                        _scope_='mmcls',
                        magnitude_key='bits',
                        magnitude_range=(
                            4,
                            0,
                        ),
                        type='Posterize'),
                    dict(
                        _scope_='mmcls',
                        magnitude_key='thr',
                        magnitude_range=(
                            256,
                            0,
                        ),
                        type='Solarize'),
                    dict(
                        _scope_='mmcls',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            110,
                        ),
                        type='SolarizeAdd'),
                    dict(
                        _scope_='mmcls',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.9,
                        ),
                        type='ColorTransform'),
                    dict(
                        _scope_='mmcls',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.9,
                        ),
                        type='Contrast'),
                    dict(
                        _scope_='mmcls',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.9,
                        ),
                        type='Brightness'),
                    dict(
                        _scope_='mmcls',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.9,
                        ),
                        type='Sharpness'),
                    dict(
                        _scope_='mmcls',
                        direction='horizontal',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.3,
                        ),
                        type='Shear'),
                    dict(
                        _scope_='mmcls',
                        direction='vertical',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.3,
                        ),
                        type='Shear'),
                    dict(
                        _scope_='mmcls',
                        direction='horizontal',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.45,
                        ),
                        type='Translate'),
                    dict(
                        _scope_='mmcls',
                        direction='vertical',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.45,
                        ),
                        type='Translate'),
                ],
                total_level=10,
                type='RandAugment'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'gt_label',
            ], type='ToTensor'),
            dict(keys=[
                'img',
                'gt_label',
            ], type='Collect'),
        ],
        type='ImageNet'),
    val=dict(
        _scope_='mmcls',
        ann_file='data/imagenet/meta/val.txt',
        data_prefix='data/imagenet/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(size=(
                236,
                -1,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='ImageNet'),
    workers_per_gpu=4)
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='accuracy/top1',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=True, type='VisualizationHook'))
default_scope = 'mmpretrain'
dist_params = dict(backend='nccl')
evaluation = dict(interval=1, metric='accuracy')
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
launcher = 'none'
load_from = None
log_config = dict(
    hooks=[
        dict(_scope_='mmcls', type='TextLoggerHook'),
    ], interval=100)
log_level = 'INFO'
lr_config = dict(
    min_lr=1e-06,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=3130,
    warmup_ratio=0.0001)
model = dict(
    backbone=dict(
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.167,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.375),
    data_preprocessor=dict(num_classes=100),
    head=dict(
        in_channels=384,
        loss=dict(
            label_smooth_val=0.1,
            loss_weight=1.0,
            mode='original',
            type='LabelSmoothLoss'),
        num_classes=100,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    train_cfg=dict(augments=[
        dict(alpha=0.2, type='Mixup'),
        dict(alpha=1.0, type='CutMix'),
    ]),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.005, type='Lamb', weight_decay=0.02),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    type='OptimWrapper')
optimizer = dict(_scope_='mmcls', lr=0.005, type='Lamb', weight_decay=0.02)
optimizer_config = dict(grad_clip=None)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.0001,
        type='LinearLR'),
    dict(
        T_max=595,
        begin=5,
        by_epoch=True,
        end=600,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
rand_increasing_policies = [
    dict(_scope_='mmcls', type='AutoContrast'),
    dict(_scope_='mmcls', type='Equalize'),
    dict(_scope_='mmcls', type='Invert'),
    dict(
        _scope_='mmcls',
        magnitude_key='angle',
        magnitude_range=(
            0,
            30,
        ),
        type='Rotate'),
    dict(
        _scope_='mmcls',
        magnitude_key='bits',
        magnitude_range=(
            4,
            0,
        ),
        type='Posterize'),
    dict(
        _scope_='mmcls',
        magnitude_key='thr',
        magnitude_range=(
            256,
            0,
        ),
        type='Solarize'),
    dict(
        _scope_='mmcls',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            110,
        ),
        type='SolarizeAdd'),
    dict(
        _scope_='mmcls',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.9,
        ),
        type='ColorTransform'),
    dict(
        _scope_='mmcls',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.9,
        ),
        type='Contrast'),
    dict(
        _scope_='mmcls',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.9,
        ),
        type='Brightness'),
    dict(
        _scope_='mmcls',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.9,
        ),
        type='Sharpness'),
    dict(
        _scope_='mmcls',
        direction='horizontal',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.3,
        ),
        type='Shear'),
    dict(
        _scope_='mmcls',
        direction='vertical',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.3,
        ),
        type='Shear'),
    dict(
        _scope_='mmcls',
        direction='horizontal',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.45,
        ),
        type='Translate'),
    dict(
        _scope_='mmcls',
        direction='vertical',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.45,
        ),
        type='Translate'),
]
resume_from = None
runner = dict(_scope_='mmcls', max_epochs=100, type='EpochBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=600)
train_dataloader = dict(
    batch_size=384,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/train.txt',
        data_prefix='train/',
        data_root='/mnt/hdd/OpenDataLab___ImageNet-100/raw/MyImagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='RepeatAugSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=384,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/val.txt',
        data_prefix='val/',
        data_root='/mnt/hdd/OpenDataLab___ImageNet-100/raw/MyImagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(name='CSPNeXt-tiny-default', project='csp'),
        type='WandbVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(name='CSPNeXt-tiny-default', project='csp'),
            type='WandbVisBackend'),
    ])
work_dir = './work_dir/cspnext-tiny-600e_in100'
workflow = [
    (
        'train',
        1,
    ),
]

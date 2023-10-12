model = dict(
    type='DINO',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer', arch='b', patch_size=16),
    neck=dict(
        type='DINONeck',
        in_channels=768,
        out_channels=65536,
        hidden_channels=2048,
        bottleneck_channels=256),
    head=dict(
        type='DINOHead',
        out_channels=65536,
        num_crops=10,
        student_temp=0.1,
        center_momentum=0.9))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='DINOMultiCrop',
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8),
    dict(type='PackInputs')
]
train_dataloader = dict(
    batch_size=32,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='mmpretrain.ImageNet',
        data_root='/data/imagenet/',
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline,
    ))
optimizer = dict(type='AdamW', lr=0.0024, betas=(0.9, 0.95), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0024, betas=(0.9, 0.95), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            ln=dict(decay_mult=0.0),
            bias=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0),
            mask_token=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0))),
    loss_scale='dynamic')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-09,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=90,
        by_epoch=True,
        begin=10,
        end=100,
        convert_to_iter_based=True)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
default_scope = 'mmpretrain'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(
    window_size=10,
    custom_cfg=[dict(data_src='', method='mean', window_size='global')])
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_level = 'INFO'
load_from = None
resume = True
randomness = dict(seed=2, diff_rank_seed=True)
custom_hooks = [
    dict(
        type='DINOTeacherTempWarmupHook',
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        teacher_temp_warmup_epochs=0,
        max_epochs=100)
]

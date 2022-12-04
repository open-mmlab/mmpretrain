dataset_type = 'ImageNet'
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            backend='memcached',
            server_list_cfg='/mnt/lustre/share/pymc/pcs_server_list.conf',
            client_cfg='/mnt/lustre/share/pymc/mc.conf',
            sys_path='/mnt/lustre/share/pymc')),
    dict(
        type='RandomResizedCrop',
        scale=448,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackClsInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            backend='petrel',
            path_mapping=dict({
                './data/WebiNat5000':
                's3://openmmlab/datasets/classification/WebiNat5000/',
                'data/WebiNat5000':
                's3://openmmlab/datasets/classification/WebiNat5000/'
            }))),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type='ImageNet',
        data_root='data/WebiNat5000/',
        ann_file='/mnt/petrelfs/share_data/liuyuan/data/testb-thr4-round3.txt',
        data_prefix='train',
        pipeline=[
            dict(type='LoadImageFromFile', ),
            dict(
                type='RandomResizedCrop',
                scale=448,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='RandAugment',
                policies='timm_increasing',
                num_policies=2,
                total_level=10,
                magnitude_level=9,
                magnitude_std=0.5,
                hparams=dict(pad_val=[104, 116, 124],
                             interpolation='bicubic')),
            dict(
                type='RandomErasing',
                erase_prob=0.25,
                mode='rand',
                min_area_ratio=0.02,
                max_area_ratio=0.3333333333333333,
                fill_color=[103.53, 116.28, 123.675],
                fill_std=[57.375, 57.12, 58.395]),
            dict(type='PackClsInputs')
        ],
        _scope_='mmcls'),
    sampler=dict(type='DefaultSampler', shuffle=True, _scope_='mmcls'))
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/WebiNat5000/',
        ann_file='/mnt/cache/liuyuan/research/draw/webinat/meta/val.txt',
        data_prefix='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=512,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=448),
            dict(type='PackClsInputs')
        ],
        _scope_='mmcls'),
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmcls'))
val_evaluator = dict(type='Accuracy', topk=(1, 5), _scope_='mmcls')
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/WebiNat5000/',
        ann_file='/mnt/cache/liuyuan/research/draw/webinat/meta/val.txt',
        data_prefix='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=512,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=448),
            dict(type='PackClsInputs')
        ],
        _scope_='mmcls'),
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmcls'))
test_evaluator = dict(type='Accuracy', topk=(1, 5), _scope_='mmcls')
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.002,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999),
        _scope_='mmcls',
        model_type='vit',
        layer_decay_rate=0.75),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })),
    constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=25,
        by_epoch=True,
        begin=5,
        end=30,
        eta_min=1e-06,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=1024)
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmcls'),
    logger=dict(type='LoggerHook', interval=100, _scope_='mmcls'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmcls'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, _scope_='mmcls', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmcls'),
    visualization=dict(
        type='VisualizationHook', enable=False, _scope_='mmcls'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend', _scope_='mmcls')]
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    _scope_='mmcls')
log_level = 'INFO'
load_from = None
resume = False
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='large',
        img_size=448,
        patch_size=16,
        drop_path_rate=0.1,
        avg_token=True,
        output_cls_token=False,
        final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='mae_vit-large-p16pt_1600.pth',  # noqa
            prefix='backbone.')),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=5000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-05)]),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8, num_classes=5000),
        dict(type='CutMix', alpha=1.0, num_classes=5000)
    ]))
file_client_args = dict(
    backend='memcached',
    path_mapping=dict({
        './data/WebiNat5000':
        's3://openmmlab/datasets/classification/WebiNat5000/',
        'data/WebiNat5000':
        's3://openmmlab/datasets/classification/WebiNat5000/'
    }),
    server_list_cfg='/mnt/lustre/share/pymc/pcs_server_list.conf',
    client_cfg='/mnt/lustre/share/pymc/mc.conf',
    sys_path='/mnt/lustre/share/pymc')
data_root = 'data/WebiNat5000/'
train_ann_file = '/mnt/cache/liuyuan/research/accv/filter_data/all_new.txt'
val_ann_file = '/mnt/cache/liuyuan/research/draw/webinat/meta/val.txt'
randomness = dict(seed=0)
val_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            backend='memcached',
            server_list_cfg='/mnt/lustre/share/pymc/pcs_server_list.conf',
            client_cfg='/mnt/lustre/share/pymc/mc.conf',
            sys_path='/mnt/lustre/share/pymc')),
    dict(
        type='ResizeEdge',
        scale=512,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackClsInputs')
]
launcher = 'slurm'
work_dir = './work_dirs/vit-large-p16_ft-8xb128-coslr-30e_in1k-448'

_base_ = 'mmpretrain::_base_/default_runtime.py'

custom_imports = dict(imports=['models'], allow_failed_imports=False)

model = dict(
    type='VideoMaskFeat',
    backbone=dict(
        type='MaskFeatMViT',
        arch='maskfeat-small',
        drop_path_rate=0.0,
        dim_mul_in_attention=False),
    neck=dict(
        type='LinearNeck',
        in_channels=768,
        out_channels=108,
        with_avg_pool=False,
        init_cfg=dict(type='TruncNormal', layer='Linear', std=0.02, bias=0)),
    head=dict(
        type='MaskFeatPretrainHead',
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    target_generator=dict(
        type='HOGGenerator3d', nbins=9, pool=8, gaussian_window=16))

# dataset settings
dataset_type = 'mmaction.VideoDataset'
data_root = 'data/kinetics400/videos_train'
ann_file_train = 'data/Kinetics400/kinetics400_train_list_videos.txt'
data_preprocessor = dict(
    type='VideoDataPreprocessor',
    mean=[114.75, 114.75, 114.75],
    std=[57.375, 57.375, 57.375],
    format_shape='NCTHW')
train_pipeline = [
    dict(type='mmaction.DecordInit'),
    dict(
        type='mmaction.SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=1),
    dict(type='mmaction.DecordDecode'),
    dict(type='mmaction.Resize', scale=(-1, 256)),
    dict(type='mmaction.RandomResizedCrop', area_range=(0.5, 1.0)),
    dict(type='mmaction.Resize', scale=(224, 224), keep_ratio=False),
    dict(type='mmaction.Flip', flip_ratio=0.5),
    dict(type='mmaction.FormatShape', input_format='NCTHW'),
    dict(
        type='MaskFeatMaskGenerator3D',
        input_size=(8, 7, 7),
        num_masking_patches=157,
        min_num_patches=9,
        max_num_patches=49),
    dict(type='PackInputs', input_key='imgs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=8e-4 * 2, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=0.02),
    paramwise_cfg=dict(
        bias_decay_mult=0.,
        norm_decay_mult=0.,
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        eta_min=1e-6,
        by_epoch=True,
        begin=10,
        end=300,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2), logger=dict(interval=100))

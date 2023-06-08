_base_ = [
    '../_base_/datasets/omnibenchmarkv2_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
# dataset settings
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_dataloader = dict(batch_size=512,     
    dataset=dict(
        data_root='data/omnibenchmarkv2/',
        ann_file='annotation/aircraft/meta/train.txt',
        data_prefix='data/aircraft/images/'),
    drop_last=True)
val_dataloader = dict(
        dataset=dict(
        data_root='data/omnibenchmarkv2/',
        ann_file='annotation/aircraft/meta/val.txt',
        data_prefix='data/aircraft/images/'),
    drop_last=False)
test_dataloader = dict(
        dataset=dict(
        data_root='data/omnibenchmarkv2/',
        ann_file='annotation/aircraft/meta/test.txt',
        data_prefix='data/aircraft/images/'),
    drop_last=False)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ViTSAM',
        arch='base',
        img_size=224,
        patch_size=16,
        out_channels=0,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        frozen_stages=12,
        out_type='avg_featmap',
        init_cfg=dict(type='Pretrained', checkpoint='/mnt/petrelfs/zhangyuanhan/weights/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth', prefix="backbone.")),
    neck=dict(type='ClsBatchNormNeck', input_features=768),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=237,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.01)]))

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='LARS', lr=6.4, weight_decay=0.0, momentum=0.9))

# learning rate scheduler
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
        T_max=80,
        by_epoch=True,
        begin=10,
        end=90,
        eta_min=0.0,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=90,val_interval=10)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=-1),
    logger=dict(type='LoggerHook', interval=10))

randomness = dict(seed=0, diff_rank_seed=True)

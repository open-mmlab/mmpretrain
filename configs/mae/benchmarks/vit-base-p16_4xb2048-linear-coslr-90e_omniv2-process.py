_base_ = [
    '../../_base_/datasets/omnibenchmarkv2_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
data_preprocessor = dict(
    num_classes=46,
)

train_dataloader = dict(batch_size=2048,     
    dataset=dict(
        data_root='data/omnibenchmarkv2/',
        ann_file='annotation/process/meta/train.txt',
        data_prefix='data/process/images/'),
    drop_last=True)
val_dataloader = dict(
        dataset=dict(
        data_root='data/omnibenchmarkv2/',
        ann_file='annotation/process/meta/val.txt',
        data_prefix='data/process/images/'),
    drop_last=False)
test_dataloader = dict(
        dataset=dict(
        data_root='data/omnibenchmarkv2/',
        ann_file='annotation/process/meta/test.txt',
        data_prefix='data/process/images/'),
    drop_last=False)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
        frozen_stages=12,
        out_type='cls_token',
        final_norm=True,
        init_cfg=dict(type='Pretrained', checkpoint='/mnt/petrelfs/zhangyuanhan/weights/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth', prefix="backbone.")),
    neck=dict(type='ClsBatchNormNeck', input_features=768),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=46,
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

# mmcls:: means we use the default settings from MMClassification
_base_ = [
    'mmcls::_base_/datasets/imagenet_bs64_swin_224.py',
    'mmcls::_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'mmcls::_base_/default_runtime.py'
]
# Fine-tuning 30 epoch is for models which have intermediate fine-tuning
# on ImageNet-21k after self-supervised pretrain.

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BEiT',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        avg_token=True,
        output_cls_token=False,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.02)]),
)

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        scale=224,
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
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(batch_size=128, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=128, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer wrapper
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999),
        model_type='vit',  # layer-wise lr decay type
        layer_decay_rate=0.75),  # layer-wise lr decay factor
    constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys={
            # the following configurations are designed for BEiTs
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            'q_bias': dict(decay_mult=0.0),
            'v_bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.gamma': dict(decay_mult=0.0),
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=20,
        end=30,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))

train_cfg = dict(by_epoch=True, max_epochs=30)

randomness = dict(seed=0)

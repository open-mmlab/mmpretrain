_base_ = [
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/default_runtime.py',
]

# dataset settings
train_dataloader = dict(batch_size=128)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MoCoV3ViT',
        arch='mocov3-small',  # embed_dim = 384
        img_size=224,
        patch_size=16,
        stop_grad_conv1=True,
        frozen_stages=12,
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.')),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(type='Normal', std=0.01, layer='Linear'),
    ))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=12, momentum=0.9, weight_decay=0.))

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=90, by_epoch=True, begin=0, end=90)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

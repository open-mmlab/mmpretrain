_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]


dist_params = dict(backend='nccl', port=16752)
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
         type='MixMIMFT',
         img_resolution=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        window_size=[14, 14, 14, 7], mlp_ratio=4, qkv_bias=True, qk_scale=None,
        drop_rate=0.0, drop_path_rate=0.1, 
        patch_norm=True, use_checkpoint=False,
        ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))


file_client_args = dict(backend='disk')
data_root = "/data/personal/nus-zwb/ImageNet/"
ann_file = "/home/nus-zwb/research/data/imagenet/meta/val.txt"

dataset_type = 'ImageNet'
preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]
val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    collate_fn=dict(type="default_collate"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
test_dataloader = val_dataloader








# runtime settings
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=100))

randomness = dict(seed=0)

_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_384.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='XCiT',
        patch_size=8,
        embed_dims=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        layer_scale_init_value=1.0,
        tokens_norm=True,
        out_type='cls_token',
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]),
)

# dataset settings
train_dataloader = dict(batch_size=128)

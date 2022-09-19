_base_ = [
    '../_base_/datasets/imagenet_bs32_pil_bicubic.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TinyViT',
        arch='tinyvit_5m_224',
        out_indices=(3, ),
        drop_path_rate=0.0,
        gap_before_final_norm=True,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
        ]),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=320,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
